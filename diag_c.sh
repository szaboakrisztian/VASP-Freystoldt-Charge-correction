#!/bin/bash
# diag_c.sh: decide whether corr's C (~2.7 V for every charge state) comes from
#   (1) mismatched pristine/defect LOCPOTs, or (2) corr's C-sampler reading the wrong region.
# usage: ./diag_c.sh <defect_dir>
set -euo pipefail

BASE=/cfs/klemming/projects/supr/qdsim/x_krsza/CN
SCRIPTS=$BASE/scripts/charge_correction
SX=$SCRIPTS/sxdefectalign
REF_DIR=$BASE/PBE/pristine
REF=$REF_DIR/cn.LOCPOT
TENSOR="7.530150,7.530150,7.222738"          # identical to the string auto_freystoldt.py passes

D=$(readlink -f "$1")
OUT=$SCRIPTS/diag_$(basename "$D")           # kept results: sx log, vline files, plot
mkdir -p "$OUT"

# ---------- 1. Parameters from the same code auto_freystoldt.py runs ----------
read DPX DPY DPZ CHG ECUT < <(SCRIPTS="$SCRIPTS" python3 - "$D" <<'PY'
import os, sys
sys.path.insert(0, os.environ["SCRIPTS"])     # make auto_freystoldt / find_defects importable
from pathlib import Path
import auto_freystoldt as af                  # import only; the __main__ guard prevents a full run
from find_defects import get_vacancies_full, read_poscar
P = Path(sys.argv[1])
cntl, _ = af.one_glob(P, "*cntl")                         # INCAR-equivalent
encut, nelect = af.parse_cntl(cntl)                       # ENCUT [eV], NELECT
geom, _ = af.one_glob(P, "*.geom_correction")             # ideal (unrelaxed) defect geometry
vac_f, vac_sp, info = get_vacancies_full(str(geom), eps=af.EPS, max_vac=af.MAX_VAC)
pgeom, _ = af.one_glob(af.Path_to_Reference.parent, "*.geom")
_, _, sp_ref, _ = read_poscar(str(pgeom))
Z = af.zval_total(sp_ref) - sum(af.VALENCE_MAP[s] for s in vac_sp)  # neutral electron count
q = int(round(Z - nelect))                                # physical charge (+ = electron deficit)
f = af.centroid_frac(vac_f, info["H"])                    # defect centre, fractional
print(f"{f[0]:.6f} {f[1]:.6f} {f[2]:.6f} {af.CHARGE_SIGN*q:d} {encut/af.RY_PER_EV:.6f}")
PY
)
echo "== params: DP=($DPX, $DPY, $DPZ)  sx --charge=$CHG (expect +1 for chg-1)  ECUT=$ECUT Ry"

# ---------- 2. Settings that change the absolute level of LOCPOT ----------
echo "== INCAR tags, pristine vs defect (missing tag = VASP default)"
for f in "$REF_DIR"/*cntl "$D"/*cntl; do
  echo "--- $f"
  grep -iE '^\s*(LVHAR|LVTOT|PREC|ENCUT|ENAUG|GGA|METAGGA|LASPH|ADDGRID|LHFCALC|LDAU|ISPIN|NELECT)\b' "$f" || echo "   (none set)"
done
echo "== POTCAR TITELs"
for p in "$REF_DIR"/POTCAR* "$D"/POTCAR*; do
  if [[ -f $p ]]; then echo "--- $p"; grep TITEL "$p"; fi
done
echo "== line 6 of the ORIGINAL pristine cn.LOCPOT (element symbols, or counts if corr already deleted it):"
sed -n 6p "$REF"

# ---------- 3. Private copies + the same VASP5 header fix corr applies ----------
W=$(mktemp -d "${TMPDIR:-/tmp}/diagc.XXXXXX"); trap 'rm -rf "$W"' EXIT
cd "$W"
copy_plain() {                                # copy or decompress $1 -> $2, original untouched
  if gzip -t "$1" 2>/dev/null; then zcat "$1" > "$2"; else cp "$1" "$2"; fi
}
copy_plain "$REF" ref.LOCPOT
SRC=$(ls "$D"/*LOCPOT*); copy_plain "$SRC" def.LOCPOT
for f in ref.LOCPOT def.LOCPOT; do
  c=$(awk 'NR==7{print substr($1,1,1); exit}' "$f")       # Direct/Cartesian on line 7 => VASP4 layout
  case "$c" in [cCdD]) ;; *) sed -i '6d' "$f" ;; esac      # else drop the element line (VASP5)
done
echo "== FFT grids (must be identical):"
for f in ref.LOCPOT def.LOCPOT; do
  printf "%s: " "$f"; awk 'g&&NF==3{print;exit} NR>7&&NF==0{g=1}' "$f"   # first line after the blank line
done

# ---------- 4. First sxdefectalign pass only (no -C), exactly as corr runs it ----------
if ! "$SX" --ecut "$ECUT" --charge "$CHG" --tensor "$TENSOR" \
     --pos "{$DPX, $DPY, $DPZ}" --relative --vref ref.LOCPOT --vdef def.LOCPOT --vasp > sx.out 2>&1; then
  cat sx.out; exit 1
fi
cp sx.out vline-eV-a*.dat "$OUT"/
echo "== sxdefectalign lines on eps/tensor/charge/correction:"
grep -iE 'eps|tensor|charge|correction' sx.out || true

# ---------- 5. Where corr samples C vs where the true far point is ----------
DPS=("$DPX" "$DPY" "$DPZ")
for a in 0 1 2; do
  awk -v dp="${DPS[$a]}" -v ax="$a" '
    NF==3 { n++; z[n]=$1; dv[n]=$2; sr[n]=$3 }    # z [bohr], dV=V_def-V_ref, V_sr=dV-V_model
    END {
      dz=z[2]-z[1]; L=z[n]-z[1]+dz                # cell length along this axis
      ff=dp+0.5; if(ff>1)ff-=1                    # corr: far point from the LINE NUMBER only
      i=int(ff*n); if(i==0)i=n
      zf=(dp+0.5)*L; while(zf>=L)zf-=L            # true far point from the z COLUMN
      for(k=1;k<=n;k++){d=z[k]-z[1]-zf; if(d<0)d=-d; if(k==1||d<bd){bd=d;b=k}}
      w=int(n/20); if(w<1)w=1                     # same +-5% window corr averages
      s=0;c=0; for(k=1;k<=n;k++){d=k-i; if(d<0)d=-d; if(d>n/2)d=n-d; if(d<=w){s+=sr[k];c++}}
      printf "axis %d: z0=%.3f L=%.3f | corr window z=%.3f -> C=%.3f | true far z=%.3f: V_sr=%.3f dV=%.3f | V_sr at defect plane z=%.3f: %.3f\n",
        ax, z[1], L, z[i], s/c, z[b], sr[b], dv[b], dp*L, sr[int(dp*n)>0?int(dp*n):1]
    }' vline-eV-a$a.dat
done

# ---------- 6. Plot (optional; skipped if matplotlib is missing) ----------
OUT="$OUT" python3 - <<'PY' || echo "(plot skipped)"
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = os.environ["OUT"]; fig, ax = plt.subplots(3, 1, figsize=(7, 9))
for a in range(3):
    rows = [l.split() for l in open(f"{out}/vline-eV-a{a}.dat") if len(l.split()) == 3]
    z, dv, sr = np.array(rows, float).T
    ax[a].plot(z, dv, label="dV = V_def - V_ref"); ax[a].plot(z, sr, label="V_sr = dV - V_model")
    ax[a].axhline(0, c="k", lw=0.5); ax[a].set_ylabel(f"axis {a} [eV]"); ax[a].legend(fontsize=8)
ax[-1].set_xlabel("z [bohr]"); fig.tight_layout(); fig.savefig(f"{out}/vsr.png", dpi=120)
print(f"plot -> {out}/vsr.png")
PY
