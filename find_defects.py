#!/usr/bin/env python3
"""Bravais translation lattice + vacancy extraction from an unrelaxed VASP POSCAR.

Criteria
--------
t in Lambda  <=>  m(t) = #{i : no same-species atom within eps of r_i + t} <= n_v_max
Lambda = HNF( Z^3  U  {accepted t} ),   S = Lambda_frac^{-1} in Z^{3x3},   M = |det S|
n_v = M * n_orbits - N   (checked)
"""

import argparse
import sys
import numpy as np
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------
# geometry primitives
# ----------------------------------------------------------------------
def shift_grid(rng=1):
    a = np.arange(-rng, rng + 1)
    return np.array(np.meshgrid(a, a, a, indexing="ij")).reshape(3, -1).T.astype(float)


def wrap(F, tiny=1e-12):
    """Fold fractional coords into [0,1) strictly (cKDTree boxsize requirement)."""
    F = np.mod(np.asarray(F, dtype=float), 1.0)
    F[F > 1.0 - tiny] = 0.0
    F[np.abs(F) < tiny] = 0.0
    return F


def min_image(dF, H, shifts):
    """True Cartesian minimum-image lengths of fractional displacements dF (n,3)."""
    dF = np.atleast_2d(np.asarray(dF, dtype=float))
    dF = dF - np.round(dF)
    cand = (dF[:, None, :] + shifts[None, :, :]) @ H
    return np.sqrt(np.einsum("ijk,ijk->ij", cand, cand).min(axis=1))


class Matcher:
    """Species-resolved periodic eps-coincidence matching with a Cartesian tolerance.

    Candidate generation uses a periodic KD-tree in fractional space with radius
    eps/sigma_min(H)  (valid superset: ||dr|| >= sigma_min ||df||);
    verification uses the exact Cartesian minimum image.
    """

    def __init__(self, F, codes, H, eps, shifts, seed=0):
        self.F = wrap(F)
        self.codes = np.asarray(codes)
        self.H = np.asarray(H, dtype=float)
        self.eps = float(eps)
        self.shifts = shifts
        sv = np.linalg.svd(self.H, compute_uv=False)
        self.smin, self.smax = float(sv.min()), float(sv.max())
        self.rf = min(0.4999, self.eps / self.smin)
        self.groups = {c: np.where(self.codes == c)[0] for c in np.unique(self.codes)}
        self.trees = {c: cKDTree(self.F[g], boxsize=1.0) for c, g in self.groups.items()}
        self.perm = np.random.default_rng(seed).permutation(len(self.F))

    def match(self, Fq, codes_q):
        """Index of the eps-coincident same-species atom for each query, -1 if none."""
        Fq = wrap(Fq)
        codes_q = np.asarray(codes_q)
        out = np.full(len(Fq), -1, dtype=int)
        for c in np.unique(codes_q):
            g = self.groups.get(c)
            if g is None:
                continue
            sel = np.where(codes_q == c)[0]
            _, j = self.trees[c].query(Fq[sel], k=1)
            gi = g[np.asarray(j, dtype=int)]
            dc = min_image(self.F[gi] - Fq[sel], self.H, self.shifts)
            out[sel] = np.where(dc < self.eps, gi, -1)
        return out

    def neighbor_lists(self):
        """All same-species eps-mates of every atom (used for orbit clustering)."""
        out = [np.empty(0, dtype=int)] * len(self.F)
        for c, g in self.groups.items():
            balls = self.trees[c].query_ball_point(self.F[g], self.rf)
            for k, cand in zip(g, balls):
                gi = g[np.asarray(cand, dtype=int)]
                d = min_image(self.F[gi] - self.F[k], self.H, self.shifts)
                out[k] = gi[d < self.eps]
        return out

    def min_nn_distance(self):
        d = []
        for c, g in self.groups.items():
            if len(g) < 2:
                continue
            df, j = self.trees[c].query(self.F[g], k=2)
            gi = g[np.asarray(j[:, 1], dtype=int)]
            d.append(min_image(self.F[gi] - self.F[g], self.H, self.shifts).min())
        return min(d) if d else np.inf


# ----------------------------------------------------------------------
# exact integer lattice algebra
# ----------------------------------------------------------------------
def row_hnf(G):
    """Upper-triangular row-echelon basis (positive diagonal) of the integer row
    lattice spanned by G. Exact: Python-int arithmetic via object dtype."""
    rows = [np.array([int(x) for x in r], dtype=object) for r in G if any(int(x) for x in r)]
    if not rows:
        raise RuntimeError("empty generator set")
    A = np.array(rows, dtype=object)
    r = 0
    for c in range(3):
        while True:
            nz = sorted([i for i in range(r, len(A)) if A[i][c] != 0], key=lambda i: abs(A[i][c]))
            if len(nz) <= 1:
                break
            p = nz[0]
            for i in nz[1:]:
                A[i] = A[i] - (A[i][c] // A[p][c]) * A[p]
        nz = [i for i in range(r, len(A)) if A[i][c] != 0]
        if not nz:
            continue
        p = nz[0]
        A[[r, p]] = A[[p, r]]
        if A[r][c] < 0:
            A[r] = -A[r]
        r += 1
    if r != 3:
        raise RuntimeError("translation lattice is rank-deficient (rank %d)" % r)
    return np.array([[int(x) for x in A[i]] for i in range(3)], dtype=np.int64)


def snap_rational(t, H, tol, qmax):
    """Smallest q <= qmax with ||(round(q t)/q - t) H|| < tol. Any t in Lambda has
    q | M <= N, so qmax = N is sufficient."""
    t = np.asarray(t, dtype=float)
    for q in range(1, int(qmax) + 1):
        n = np.round(q * t)
        if np.linalg.norm((n / q - t) @ H) < tol:
            return n.astype(np.int64), q
    return None, None


def build_lattice(acc, H, tol, qmax):
    """Lambda_frac (rows = primitive vectors in H-fractional coords), S, M."""
    D = 1
    ints = []
    for t in acc:
        n, q = snap_rational(t, H, tol, qmax)
        if n is None:
            continue
        ints.append((n, q))
        D = int(np.lcm(D, q))
    rows = [(D * np.eye(3, dtype=np.int64))[i] for i in range(3)]
    rows += [n * (D // q) for n, q in ints]
    Lam_frac = row_hnf(rows).astype(float) / D
    S = np.linalg.inv(Lam_frac)
    if not np.allclose(S, np.round(S), atol=1e-6):
        raise RuntimeError("S = Lambda^{-1} not integral:\n%s" % S)
    S = np.round(S).astype(np.int64)
    M = int(round(abs(np.linalg.det(S.astype(float)))))
    if M < 1:
        raise RuntimeError("non-positive supercell index M")
    return Lam_frac, S, M


def in_lattice(t, Lam_frac, H, tol):
    c = np.asarray(t, dtype=float) @ np.linalg.inv(Lam_frac)
    return np.linalg.norm((c - np.round(c)) @ Lam_frac @ H) < tol


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def read_poscar(filepath):
    with open(filepath, "r") as f:
        lines = [ln for ln in f.read().splitlines()]

    H0 = np.array([list(map(float, lines[i].split()[:3])) for i in range(2, 5)])
    s = float(lines[1].split()[0])
    scale = s if s > 0.0 else (abs(s) / abs(np.linalg.det(H0))) ** (1.0 / 3.0)
    H = H0 * scale
    if abs(np.linalg.det(H)) < 1e-8:
        raise ValueError("singular cell matrix")

    try:  # VASP 4 format: no species line
        counts = list(map(int, lines[5].split()))
        symbols = ["X%d" % k for k in range(len(counts))]
        off = 6
    except ValueError:
        symbols = lines[5].split()
        counts = list(map(int, lines[6].split()))
        off = 7
    if len(symbols) != len(counts):
        raise ValueError("species/count mismatch")

    N = int(sum(counts))
    species = np.repeat(symbols, counts)

    flag = lines[off].strip().lower()
    if flag.startswith("s"):
        flag = lines[off + 1].strip().lower()
        start = off + 2
    else:
        start = off + 1

    raw = np.array([list(map(float, lines[i].split()[:3])) for i in range(start, start + N)])
    if flag.startswith("d"):
        R = raw @ H
    elif flag[0] in ("c", "k"):
        R = raw * scale
    else:
        raise ValueError("unrecognised coordinate mode: %r" % flag)

    _, codes = np.unique(species, return_inverse=True)
    return H, R, species, np.asarray(codes).ravel()


# ----------------------------------------------------------------------
# stage 1: translation lattice
# ----------------------------------------------------------------------
def find_lattice(F, codes, H, eps, shifts, max_vac, n_ref=4, tmax=None, verbose=True):
    N = len(F)
    mm = Matcher(F, codes, H, eps, shifts)
    dmin = mm.min_nn_distance()
    if eps >= dmin * mm.smin / mm.smax:
        print("[!] eps=%.3g too large: requires eps < d_min*smin/smax = %.3g"
              % (eps, dmin * mm.smin / mm.smax), file=sys.stderr)

    vals, cnts = np.unique(codes, return_counts=True)
    c_major = vals[int(np.argmax(cnts))]
    ix = np.where(codes == c_major)[0]
    refs = ix[: max(1, n_ref)]
    cand = wrap(np.vstack([F[ix] - F[i0] for i0 in refs]))
    cand = np.unique(np.round(cand, 8), axis=0)

    L = min_image(cand, H, shifts)
    keep = L > eps
    cand, L = cand[keep], L[keep]
    if tmax is not None:
        keep = L <= tmax
        cand, L = cand[keep], L[keep]
    order = np.argsort(L)
    cand, L = cand[order], L[order]

    stages = sorted(set([min(16, N), min(64, N), min(256, N), N]))
    Lam_frac, S, M = np.eye(3), np.eye(3, dtype=np.int64), 1
    acc = []

    def misses(t):
        """Progressive-subset evaluation of m(t); subset misses <= m(t) so no
        false rejection can occur."""
        m = 0
        for s in stages:
            sub = mm.perm[:s]
            m = int((mm.match(F[sub] + t, codes[sub]) < 0).sum())
            if m > max_vac:
                return m, False
        return m, True

    for t, ell in zip(cand, L):
        if in_lattice(t, Lam_frac, H, eps):
            continue
        mf, ok = misses(t)
        if not ok:
            continue
        mb, ok = misses(-t)
        if not ok:
            continue
        acc.append(t)
        Lam_frac, S, M = build_lattice(acc, H, eps, qmax=N)
        if verbose:
            print("    + t = (%8.5f %8.5f %8.5f)  |t| = %7.4f A  m(+t)=%d m(-t)=%d  -> M = %d"
                  % (t[0], t[1], t[2], ell, mf, mb, M))
        if M >= N:  # cannot exceed N sites per orbit
            break

    return Lam_frac, S, M, np.array(acc).reshape(-1, 3), mm


# ----------------------------------------------------------------------
# stage 2: orbits + vacancies
# ----------------------------------------------------------------------
def coset_reps(S):
    """Representatives of Z^3 / rowlattice(S); exactly |det S| of them.
    (f -> f + m, m in Z^3, corresponds to g -> g + m S in primitive coords.)"""
    T = row_hnf(S)
    a = np.arange(T[0, 0])
    b = np.arange(T[1, 1])
    c = np.arange(T[2, 2])
    return np.array(np.meshgrid(a, b, c, indexing="ij")).reshape(3, -1).T.astype(float)


def find_vacancies(F, codes, H, Lam_frac, S, M, eps, shifts, sup=None):
    if sup is None:
        sup = Matcher(F, codes, H, eps, shifts)
    Lam_cart = Lam_frac @ H
    G = wrap(F @ np.linalg.inv(Lam_frac))
    prim = Matcher(G, codes, Lam_cart, eps, shifts)

    # --- orbit decomposition by eps-coincidence in the primitive cell (periodic) ---
    nbr = prim.neighbor_lists()
    label = np.full(len(F), -1, dtype=int)
    reps = []
    for i in range(len(F)):
        if label[i] >= 0:
            continue
        mates = np.asarray(nbr[i], dtype=int)
        prev = label[mates]
        if np.any(prev >= 0):
            raise RuntimeError("non-transitive clustering at atom %d: reduce eps" % i)
        label[mates] = len(reps)
        reps.append(i)

    reps_n = coset_reps(S)
    if len(reps_n) != M:
        raise RuntimeError("coset enumeration gave %d != M = %d" % (len(reps_n), M))

    vac_f, vac_code = [], []
    for o, i0 in enumerate(reps):
        occ = int((label == o).sum())
        if occ > M:
            raise RuntimeError("orbit %d: occupancy %d > M = %d (eps too large)" % (o, occ, M))
        Fe = wrap((G[i0] + reps_n) @ Lam_frac)
        miss = sup.match(Fe, np.full(len(Fe), codes[i0])) < 0
        vac_f.extend(Fe[miss])
        vac_code.extend([codes[i0]] * int(miss.sum()))

    n_expect = M * len(reps) - len(F)
    if len(vac_f) != n_expect:
        raise RuntimeError("inconsistent count: enumerated %d, M*n_orb-N = %d"
                           % (len(vac_f), n_expect))
    return np.array(vac_f).reshape(-1, 3), np.asarray(vac_code, dtype=int), len(reps)


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------
def get_vacancies_full(filepath, eps=1e-4, max_vac=None, n_ref=4,
                       shift_range=1, verbose=False):
    """Returns (vac_frac (n,3), vac_species (n,), info dict)."""
    H, R, species, codes = read_poscar(filepath)
    shifts = shift_grid(shift_range)
    F = wrap(R @ np.linalg.inv(H))
    N = len(F)
    if max_vac is None:
        max_vac = max(1, N // 20)

    Lam_frac, S, M, acc, sup = find_lattice(F, codes, H, eps, shifts, max_vac,
                                            n_ref=n_ref, verbose=verbose)
    info = dict(H=H, Lam_frac=Lam_frac, Lam_cart=Lam_frac @ H, S=S, M=M,
                N=N, n_orb=N, degenerate=(M == 1))
    if M == 1:
        return np.empty((0, 3)), np.empty(0, dtype="<U8"), info

    vac_f, vac_code, n_orb = find_vacancies(F, codes, H, Lam_frac, S, M,
                                            eps, shifts, sup=sup)
    info["n_orb"] = n_orb
    sym = np.unique(species)
    return vac_f, np.asarray([sym[c] for c in vac_code], dtype="<U8"), info


def get_vacancies(filepath, eps=1e-4, decimals=5, strict=True, **kw):
    """Legacy contract: fractional vacancy coords only, ndarray shape (n_v, 3).

    Wrapped to [0,1), rounded to `decimals`, lexicographically sorted so the row
    order is deterministic and independent of atom ordering in the POSCAR.
    Empty result is shape (0, 3) -- len() == 0, .shape[1] == 3, iteration safe.

    strict=True raises when M == 1 (no translational sublattice => the vacancy
    set is undefined, not empty). Set strict=False to return an empty array.
    """
    vac_f, _, info = get_vacancies_full(filepath, eps=eps, **kw)

    if info["degenerate"]:
        if strict:
            raise RuntimeError(
                "M = 1 for %s: primitive cell == supercell, no reference "
                "sublattice, vacancy set undefined" % filepath)
        return np.empty((0, 3))

    if len(vac_f) == 0:
        return np.empty((0, 3))

    out = np.mod(np.round(np.asarray(vac_f, dtype=float), decimals), 1.0)
    return out[np.lexsort((out[:, 2], out[:, 1], out[:, 0]))]


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Bravais basis + vacancy extraction (ideal POSCAR).")
    p.add_argument("poscar", type=str)
    p.add_argument("--eps", type=float, default=1e-4,
                   help="Cartesian coincidence tolerance in Angstrom (default 1e-4)")
    p.add_argument("--max-vac", type=int, default=None,
                   help="upper bound n_v_max used in the criterion m(t) <= n_v_max")
    p.add_argument("--n-ref", type=int, default=4,
                   help="reference atoms per majority species for candidate generation")
    p.add_argument("--shift-range", type=int, default=1,
                   help="minimum-image shell half-width (increase for strongly skewed cells)")
    args = p.parse_args()

    print("[*] Parsing geometry: %s" % args.poscar)
    try:
        H, R, species, codes = read_poscar(args.poscar)
    except Exception as e:
        print("[!] Parsing failed: %s" % e)
        sys.exit(1)

    N = len(R)
    shifts = shift_grid(args.shift_range)
    F = wrap(R @ np.linalg.inv(H))
    max_vac = args.max_vac if args.max_vac is not None else max(1, N // 20)
    print("[*] N = %d atoms, species %s, V = %.4f A^3, n_v_max = %d"
          % (N, dict(zip(*np.unique(species, return_counts=True))),
             abs(np.linalg.det(H)), max_vac))

    print("[*] Searching translation lattice:")
    try:
        Lam_frac, S, M, acc, sup = find_lattice(F, codes, H, args.eps, shifts, max_vac,
                                                n_ref=args.n_ref, verbose=True)
    except Exception as e:
        print("[!] Lattice determination failed: %s" % e)
        sys.exit(1)

    Lam_cart = Lam_frac @ H
    print("[*] Primitive vectors (Cartesian, A):")
    for i in range(3):
        print("      a_%d = (%9.5f %9.5f %9.5f)   |a_%d| = %8.5f"
              % (i + 1, Lam_cart[i, 0], Lam_cart[i, 1], Lam_cart[i, 2], i + 1,
                 np.linalg.norm(Lam_cart[i])))
    print("[*] S = H/Lambda (integral) =\n%s\n    M = |det S| = %d" % (S, M))

    if M == 1:
        print("[!] Primitive cell == supercell: no internal translational symmetry, "
              "vacancy detection is vacuous (no reference sublattice).")
        sys.exit(0)

    try:
        vac_f, vac_code, n_orb = find_vacancies(F, codes, H, Lam_frac, S, M,
                                                args.eps, shifts, sup=sup)
    except Exception as e:
        print("[!] Vacancy mapping failed: %s" % e)
        sys.exit(1)

    sym = np.unique(species)
    print("[*] Orbits n_orb = %d,  invariant n_v = M*n_orb - N = %d*%d - %d = %d"
          % (n_orb, M, n_orb, N, M * n_orb - N))
    if len(vac_f) == 0:
        print("[+] Topological mapping complete. Zero vacancies detected.")
    else:
        print("[!] Symmetry broken. Detected %d vacancy defect(s):" % len(vac_f))
        for i, (f, c) in enumerate(zip(vac_f, vac_code)):
            v = f @ H
            print("    V_%d [%s]: Cartesian(%9.4f, %9.4f, %9.4f) | Fractional(%7.4f, %7.4f, %7.4f)"
                  % (i + 1, sym[c], v[0], v[1], v[2], f[0], f[1], f[2]))
    print("[i] Note: an orbit with all M members removed is invisible to this method "
          "(requires a reference ideal structure).")


if __name__ == "__main__":
    main()
