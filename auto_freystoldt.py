#!/usr/bin/env python3
"""Freysoldt (sxdefectalign) charge corrections over a defect directory tree.

Charge bookkeeping
------------------
    Z_d = Z_pristine - sum_v ZVAL[s(v)]     neutral electron count of the defect cell
    q   = Z_d - NELECT                       physical charge (electron deficit > 0)
    CHARGE in corr.par = -q                  sxdefectalign electron-count convention

The `corr` wrapper owns the sxdefectalign invocation (--vasp, --relative,
--ecut [Ry], the -C alignment pass). This script builds corr.par, runs the
wrapper once per directory, parses the correction, and leaves NO residue:

  * corr runs in a private scratch dir (deleted afterwards, whatever it wrote);
  * gzipped LOCPOTs are expanded into scratch, the .gz is left untouched;
  * VREF points at the reference LOCPOT in place -- nothing is copied;
  * each data dir is snapshotted before/after; new entries are removed;
  * atexit + SIGTERM/SIGINT/SIGHUP handlers purge on interruption;
  * a startup sweep removes artefacts left by SIGKILLed previous runs.

DPx/DPy/DPz are FRACTIONAL (corr passes --relative).
Requires ideal (unrelaxed) *.geom files: the lattice extractor uses eps = 1e-4 A.
"""

import atexit
import gzip
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from find_defects import get_vacancies_full, read_poscar, shift_grid

try:  # survive truncated batch-job logs
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass

# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
Path_to_Charges = Path("/cfs/klemming/projects/supr/qdsim/x_krsza/CN/PBE/defects/double_vacancies/carbon2-nitrogen").resolve()
Path_to_Reference = Path("/cfs/klemming/projects/supr/qdsim/x_krsza/CN/PBE/pristine/carbon2-nitrogen/cn.LOCPOT").resolve()
Output_Path = Path("/cfs/klemming/projects/supr/qdsim/x_krsza/CN/scripts/charge_correction/carbon2-nitrogen.dat").resolve()
CORR_EXECUTABLE = Path("/cfs/klemming/projects/supr/qdsim/x_krsza/CN/scripts/charge_correction/corr").resolve()

# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------
# POTCAR ZVAL, not chemical valence. Verify against the POTCAR headers used.
VALENCE_MAP = {"H": 1, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
               "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ga": 3, "As": 5}

RY_PER_EV = 13.605693009
EPS = 1e-4            # Cartesian coincidence tolerance [A]; ideal geometries only
MAX_VAC = 8           # n_v_max in the lattice criterion  m(t) <= n_v_max
CHARGE_SIGN = -1      # sxdefectalign: --charge = -q
VOL_TOL = 1e-3        # |V_def/V_ref - 1| tolerance

SCRATCH_ROOT = None   # None -> $SNIC_TMP / $TMPDIR / system temp
TMP_PREFIX = ".corrwrk-"
SYMLINK_VREF_IN_CWD = False   # True only if corr hardcodes ./REF_LOCPOT
FAILED_LOG_DIR = Output_Path.parent / "corr_failed_logs"   # outside the data tree
KEEP_FAILED_LOGS = True       # captures of runs whose output could not be parsed

# artefacts from earlier versions of this script / killed runs
STALE_NAMES = ("REF_LOCPOT", "corr.par", "corr.stdout.log")
STALE_GLOBS = ("vline-eV-a[0-9]*.dat", "REF_LOCPOT.*", TMP_PREFIX + "*")
# never delete these, whatever the snapshot says
PROTECTED_RX = re.compile(r"(\.geom$|cntl|LOCPOT|OUTCAR|CONTCAR|POSCAR|INCAR|"
                          r"KPOINTS|POTCAR|WAVECAR|CHGCAR|vasprun|\.gz$|\.xml$)")

_NUM = r"[-+]?\d+\.?\d*(?:[eEdD][-+]?\d+)?"
CORR_PATTERNS = (
    r"Defect\s+correction\s*\(eV\)\s*[:=]?\s*(" + _NUM + r")",
    r"Difference\s*\(eV\)\s*[:=]?\s*(" + _NUM + r")",
    r"[Tt]otal\s+correction\s*(?:\(eV\))?\s*[:=]?\s*(" + _NUM + r")",
    r"[Cc]orrection[^\n]*?[:=]\s*(" + _NUM + r")",
)
CORR_RX = [re.compile(p) for p in CORR_PATTERNS]
RESULT_FILE_GLOBS = ("*.log", "*.out", "*.txt", "*.dat", "corr*", "*result*")
RESULT_FILE_SKIP = ("corr.par",)

# ----------------------------------------------------------------------
# crash-proof cleanup registry
# ----------------------------------------------------------------------
_ACTIVE = []          # paths (files or dirs) to remove if we die mid-iteration


def _rm(path):
    try:
        p = Path(path)
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    except OSError:
        pass


def _register(path):
    _ACTIVE.append(Path(path))
    return Path(path)


def _unregister(path):
    try:
        _ACTIVE.remove(Path(path))
    except ValueError:
        pass


def _purge(*_a):
    while _ACTIVE:
        _rm(_ACTIVE.pop())


def _signal_exit(signum, _frame):
    _purge()
    sys.stderr.write("\n[DEBUG] caught signal %d: scratch purged, exiting\n" % signum)
    sys.exit(128 + signum)


atexit.register(_purge)
for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    try:
        signal.signal(_s, _signal_exit)
    except (ValueError, OSError, AttributeError):
        pass


# ----------------------------------------------------------------------
# filesystem helpers
# ----------------------------------------------------------------------
def scratch_base():
    if SCRATCH_ROOT:
        return Path(SCRATCH_ROOT)
    for var in ("SNIC_TMP", "TMPDIR", "SLURM_TMPDIR"):
        v = os.environ.get(var)
        if v and Path(v).is_dir():
            return Path(v)
    return Path(tempfile.gettempdir())


def snapshot(directory):
    """Names of every entry currently in `directory` (non-recursive)."""
    try:
        return {e.name for e in directory.iterdir()}
    except OSError:
        return set()


def remove_new_entries(directory, before, label=""):
    """Delete entries that appeared since `before`, honouring PROTECTED_RX."""
    removed, kept = [], []
    try:
        entries = list(directory.iterdir())
    except OSError:
        return removed
    for e in entries:
        if e.name in before:
            continue
        if PROTECTED_RX.search(e.name):
            kept.append(e.name)
            continue
        _rm(e)
        removed.append(e.name)
    if removed:
        print("[DEBUG] %scleaned %d new artefact(s): %s"
              % (label, len(removed), ", ".join(sorted(removed))))
    if kept:
        print("[DEBUG] %skept protected new file(s): %s" % (label, ", ".join(sorted(kept))))
    return removed


def sweep_stale(root):
    """Remove residue from killed runs: REF_LOCPOT copies, corr.par, vline dumps,
    orphaned scratch dirs."""
    n = 0
    for d in [root, *root.rglob("*")]:
        if not d.is_dir():
            continue
        cands = [d / x for x in STALE_NAMES]
        for g in STALE_GLOBS:
            cands.extend(d.glob(g))
        for f in set(cands):
            if f.exists() or f.is_symlink():
                _rm(f)
                n += 1
    base = scratch_base()
    try:
        for d in base.glob(TMP_PREFIX + "*"):
            if d.is_dir() and d.stat().st_mtime < time.time() - 3600:
                _rm(d)
                n += 1
    except OSError:
        pass
    if n:
        print("[DEBUG] startup sweep removed %d stale artefact(s)" % n)
    return n


def is_gzipped(path):
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def one_glob(directory, pattern, exclude=()):
    """Exactly-one-match glob. Returns (path_or_None, all_hits)."""
    hits = sorted(x for x in directory.glob(pattern)
                  if x.is_file() and not any(e in x.name for e in exclude))
    return (hits[0] if len(hits) == 1 else None), hits


def expand_gz(src, dst):
    """Stream-decompress src -> dst. The original .gz is left untouched."""
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=1 << 22)
    return dst


# ----------------------------------------------------------------------
# physics-side helpers
# ----------------------------------------------------------------------
def zval_total(species):
    sym, cnt = np.unique(np.asarray(species), return_counts=True)
    missing = [s for s in sym if s not in VALENCE_MAP]
    if missing:
        raise KeyError("no ZVAL entry for %s" % missing)
    return int(sum(VALENCE_MAP[s] * int(n) for s, n in zip(sym, cnt)))


def parse_cntl(path):
    """Extract (ENCUT [eV], NELECT). Tolerates quotes, comments, Fortran exponents."""
    out = {"ENCUT": None, "NELECT": None}
    rx = {k: re.compile(r"^\s*%s\s*=\s*[\"']?\s*(" % k + _NUM + r")", re.IGNORECASE)
          for k in out}
    with open(path) as f:
        for line in f:
            s = line.split("#")[0].split("!")[0]
            for k, r in rx.items():
                m = r.search(s)
                if m:
                    out[k] = float(m.group(1).replace("D", "E").replace("d", "e"))
    return out["ENCUT"], out["NELECT"]


def centroid_frac(vac_f, H):
    """Circular mean of vacancy sites via true Cartesian minimum image about vac_f[0]."""
    vac_f = np.atleast_2d(np.asarray(vac_f, dtype=float))
    if len(vac_f) == 1:
        return np.mod(vac_f[0], 1.0)
    sh = shift_grid(1)
    d = vac_f - vac_f[0]
    d -= np.round(d)
    cand = d[:, None, :] + sh[None, :, :]
    cart = cand @ H
    k = np.argmin(np.einsum("ijk,ijk->ij", cart, cart), axis=1)
    d = cand[np.arange(len(cand)), k]
    return np.mod(vac_f[0] + d.mean(axis=0), 1.0)


def max_pair_separation(vac_f, H):
    """Largest minimum-image vacancy-vacancy distance [A]; charge-model sanity check."""
    vac_f = np.atleast_2d(np.asarray(vac_f, dtype=float))
    if len(vac_f) < 2:
        return 0.0
    sh = shift_grid(1)
    dmax = 0.0
    for i in range(len(vac_f)):
        d = vac_f[i + 1:] - vac_f[i]
        if not len(d):
            continue
        d -= np.round(d)
        cart = (d[:, None, :] + sh[None, :, :]) @ H
        dmax = max(dmax, float(np.sqrt(np.einsum("ijk,ijk->ij", cart, cart).min(axis=1)).max()))
    return dmax


def write_par(path, encut_eV, charge_par, tensor, f_center, vref, vdef):
    """corr.par. ECUTOFF in Ry; DP* fractional -- corr passes --relative."""
    with open(path, "w") as f:
        f.write('ECUTOFF="%.6f"\n' % (encut_eV / RY_PER_EV))
        f.write('CHARGE="%d"\n' % charge_par)
        f.write('TENSOR="%s"\n' % tensor)
        f.write('DPx="%.6f"\n' % f_center[0])
        f.write('DPy="%.6f"\n' % f_center[1])
        f.write('DPz="%.6f"\n' % f_center[2])
        f.write('VREF="%s"\n' % Path(vref).resolve())
        f.write('VDEF="%s"\n' % Path(vdef).resolve())


def _scan_text(text):
    for pi, rx in enumerate(CORR_RX):
        hits = rx.findall(text or "")
        if hits:
            try:
                return float(hits[-1].replace("D", "E").replace("d", "e")), pi
            except ValueError:
                continue
    return None, None


def parse_correction(stdout, stderr="", workdir=None):
    """stdout, then stderr, then any result file corr left in the scratch dir.
    Returns (value, pattern_index, source) or (None, None, None)."""
    for label, text in (("stdout", stdout), ("stderr", stderr)):
        v, pi = _scan_text(text)
        if v is not None:
            return v, pi, label
    if workdir is None:
        return None, None, None
    cands = set()
    for g in RESULT_FILE_GLOBS:
        for f in workdir.glob(g):
            if (f.is_file() and f.name not in RESULT_FILE_SKIP
                    and "LOCPOT" not in f.name and f.stat().st_size < (8 << 20)):
                cands.add(f)
    for f in sorted(cands, key=lambda p: -p.stat().st_mtime):
        try:
            v, pi = _scan_text(f.read_text(errors="replace"))
        except OSError:
            continue
        if v is not None:
            return v, pi, "file:%s" % f.name
    return None, None, None


def emit(dir_path, q, value):
    """Append one result row; never let an I/O error abort the scan."""
    row = "%s    %+d    %.6f\n" % (dir_path, q, value)
    try:
        with open(Output_Path, "a") as out:
            out.write(row)
            out.flush()
        return True
    except OSError as e:
        print("[ERROR] cannot append to %s (%s)" % (Output_Path, e))
        print("[ERROR] ROW: %s" % row.strip())
        return False


def save_failed_log(P, cmd, res):
    if not KEEP_FAILED_LOGS:
        return None
    try:
        FAILED_LOG_DIR.mkdir(parents=True, exist_ok=True)
        tag = str(P).strip("/").replace("/", "_")[-120:]
        f = FAILED_LOG_DIR / ("%s.log" % tag)
        f.write_text("$ %s   (cwd=%s)\n--- STDOUT ---\n%s\n--- STDERR ---\n%s\n"
                     % (cmd, P, res.stdout, res.stderr))
        return f
    except OSError:
        return None


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def Run_corrections(folders, tensor, fresh=True):
    root = Path(folders).resolve()
    Output_Path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and Output_Path.exists():
        bak = Output_Path.with_suffix(Output_Path.suffix + ".bak")
        Output_Path.replace(bak)
        print("[DEBUG] previous output moved to %s" % bak)

    if not CORR_EXECUTABLE.exists():
        print("[FATAL] corr executable not found: %s" % CORR_EXECUTABLE)
        sys.exit(1)
    if not Path_to_Reference.exists():
        print("[FATAL] reference LOCPOT not found: %s" % Path_to_Reference)
        sys.exit(1)
    if not root.is_dir():
        print("[FATAL] scan root is not a directory: %s" % root)
        sys.exit(1)

    sweep_stale(root)
    print("[DEBUG] scratch base: %s" % scratch_base())

    # ---- pristine reference ----
    ref_dir = Path_to_Reference.parent
    pristine_geom, hits = one_glob(ref_dir, "*.geom")
    if pristine_geom is None:
        print("[FATAL] expected exactly one *.geom in %s, found %s"
              % (ref_dir, [h.name for h in hits]))
        sys.exit(1)
    try:
        H_ref, R_ref, sp_ref, _ = read_poscar(str(pristine_geom))
        Z_pristine = zval_total(sp_ref)
    except Exception as e:
        print("[FATAL] pristine reference unusable (%s): %s" % (pristine_geom, e))
        sys.exit(1)
    V_ref = abs(np.linalg.det(H_ref))
    print("[DEBUG] pristine %s: N=%d, Z=%d, V=%.4f A^3"
          % (pristine_geom.name, len(R_ref), Z_pristine, V_ref))
    print("[DEBUG] scanning: %s" % root)

    dirs = sorted({d for d in [root, *root.rglob("*")] if d.is_dir()}, reverse=True)
    print("[DEBUG] %d candidate directory(ies)" % len(dirs))
    n_done = n_skip = 0

    for P in dirs:
        # ---- LOCPOT ----
        locpot_src, hits = one_glob(P, "*LOCPOT*", exclude=("REF_LOCPOT",))
        if locpot_src is None:
            if hits:
                print("[DEBUG] SKIP %s: ambiguous LOCPOT set %s" % (P, [h.name for h in hits]))
                n_skip += 1
            continue

        print("\n[DEBUG] --------------------------------------------------")
        print("[DEBUG] Target: %s" % P)
        print("[DEBUG] LOCPOT: %s" % locpot_src.name)

        # ---- cntl ----
        cntl_file, hits = one_glob(P, "*cntl")
        if cntl_file is None:
            print("[DEBUG] SKIP %s: missing/ambiguous *cntl %s"
                  % (P.name, [h.name for h in hits]))
            n_skip += 1
            continue
        encut, nelect = parse_cntl(cntl_file)
        if encut is None:
            print("[DEBUG] SKIP %s: ENCUT absent from %s" % (P.name, cntl_file.name))
            n_skip += 1
            continue

        # ---- defect geometry ----
        defect_geom, hits = one_glob(P, "*.geom_correction")
        if defect_geom is None:
            print("[DEBUG] SKIP %s: expected one *.geom_correction, found %s"
                  % (P.name, [h.name for h in hits]))
            n_skip += 1
            continue

        # ---- vacancy extraction (species-resolved) ----
        try:
            vac_f, vac_sp, info = get_vacancies_full(str(defect_geom), eps=EPS, max_vac=MAX_VAC)
        except Exception as e:
            print("[DEBUG] SKIP %s: lattice extraction failed: %s" % (P.name, e))
            n_skip += 1
            continue
        if info["degenerate"]:
            print("[DEBUG] SKIP %s: M = 1, no reference sublattice" % P.name)
            n_skip += 1
            continue
        if len(vac_f) == 0:
            print("[DEBUG] SKIP %s: no vacancy detected (or an entire orbit is absent)" % P.name)
            n_skip += 1
            continue

        H_def = info["H"]
        r_vol = abs(np.linalg.det(H_def)) / V_ref
        if abs(r_vol - 1.0) > VOL_TOL:
            print("[DEBUG] SKIP %s: V_def/V_ref = %.6f != 1 (LOCPOT grids incommensurate)"
                  % (P.name, r_vol))
            n_skip += 1
            continue

        # ---- charge state ----
        try:
            _, _, sp_def, _ = read_poscar(str(defect_geom))
            Z_vac = int(sum(VALENCE_MAP[s] for s in vac_sp))
            Z_cell = Z_pristine - Z_vac
            Z_direct = zval_total(sp_def)
        except KeyError as e:
            print("[DEBUG] SKIP %s: %s" % (P.name, e))
            n_skip += 1
            continue
        if Z_cell != Z_direct:
            print("[DEBUG] SKIP %s: Z mismatch, pristine-vac=%d vs geom=%d "
                  "(wrong reference cell, or geometry is relaxed)"
                  % (P.name, Z_cell, Z_direct))
            n_skip += 1
            continue

        q = 0 if nelect is None else int(round(Z_cell - nelect))
        if nelect is None:
            print("[DEBUG] NELECT absent in %s -> assuming neutral q = 0" % cntl_file.name)
        print("[DEBUG] n_vac=%d %s | M=%d n_orb=%d | Z_cell=%d NELECT=%s -> q=%+d (CHARGE=%+d)"
              % (len(vac_f), list(vac_sp), info["M"], info["n_orb"], Z_cell,
                 "None" if nelect is None else int(nelect), q, CHARGE_SIGN * q))

        f_center = centroid_frac(vac_f, H_def)
        cart = f_center @ H_def
        print("[DEBUG] center frac (%.6f %.6f %.6f) -> cart (%.4f %.4f %.4f) A"
              % (f_center[0], f_center[1], f_center[2], cart[0], cart[1], cart[2]))
        if len(vac_f) > 1:
            print("[DEBUG] multi-vacancy: max separation %.3f A; a single Gaussian at the "
                  "centroid is valid only if this is <~ beta"
                  % max_pair_separation(vac_f, H_def))

        # ---- run corr in private scratch; snapshot P as a second safety net ----
        before = snapshot(P)
        workdir = None
        try:
            workdir = _register(Path(tempfile.mkdtemp(prefix=TMP_PREFIX, dir=str(scratch_base()))))

            if is_gzipped(locpot_src):
                vdef = workdir / locpot_src.name.replace(".gz", "")
                print("[DEBUG] expanding %s -> scratch (original .gz untouched)"
                      % locpot_src.name)
                expand_gz(locpot_src, vdef)
            else:
                vdef = locpot_src

            vref = Path_to_Reference
            if SYMLINK_VREF_IN_CWD:
                link = workdir / "REF_LOCPOT"
                try:
                    link.symlink_to(Path_to_Reference)
                except OSError:
                    shutil.copy(Path_to_Reference, link)
                vref = link

            param_path = workdir / "corr.par"
            write_par(param_path, encut, CHARGE_SIGN * q, tensor, f_center, vref, vdef)

            res = subprocess.run([str(CORR_EXECUTABLE), "-p", param_path.name],
                                 cwd=str(workdir), capture_output=True, text=True, check=True)

        except subprocess.CalledProcessError as e:
            print("[ERROR] corr failed in %s (exit %d)\nSTDOUT:\n%s\nSTDERR:\n%s"
                  % (P, e.returncode, e.stdout, e.stderr))
            n_skip += 1
            continue
        except OSError as e:
            print("[ERROR] I/O failure for %s: %s" % (P, e))
            n_skip += 1
            continue
        else:
            final_corr, pi, src = parse_correction(res.stdout, res.stderr, workdir=workdir)
            if final_corr is None:
                logf = save_failed_log(P, "%s -p corr.par" % CORR_EXECUTABLE, res)
                print("[DEBUG] %s: corr exited 0 but no correction parsed "
                      "(stdout %d B, stderr %d B)%s"
                      % (P.name, len(res.stdout or ""), len(res.stderr or ""),
                         "" if logf is None else "; capture -> %s" % logf))
                print("[DEBUG]   scratch contents: %s"
                      % sorted(f.name for f in workdir.iterdir()))
                if not (res.stdout or "").strip():
                    print("[DEBUG]   stdout EMPTY -> corr redirects its result; "
                          "grep -nE 'tee|>|>>|1>&2' %s" % CORR_EXECUTABLE)
                n_skip += 1
                continue

            if pi != 0 or src != "stdout":
                print("[DEBUG]   matched pattern %d on %s" % (pi, src))
            print("Parent: %s | Charge: %+d -> Correction: %.6f eV" % (P.parent, q, final_corr))
            if emit(P, q, final_corr):
                n_done += 1
            else:
                n_skip += 1
        finally:
            if workdir is not None:
                _unregister(workdir)
                _rm(workdir)
            remove_new_entries(P, before, label="%s: " % P.name)

    print("\n[DEBUG] done: %d correction(s), %d skipped -> %s" % (n_done, n_skip, Output_Path))
    if n_done == 0:
        print("[DEBUG] nothing written. Read the per-directory SKIP/ERROR lines above; "
              "if corr ran but nothing parsed, inspect %s and adjust CORR_PATTERNS[0] "
              "to the literal output line." % FAILED_LOG_DIR)


if __name__ == "__main__":
    Run_corrections(folders=Path_to_Charges,
                    tensor="7.530150,7.530150,7.222738")
