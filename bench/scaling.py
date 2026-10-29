"""Scaling benchmark: Si supercells vs ABACUS (1-core).

Generates Si supercells from tests/regress/si-gamma, runs ABACUS directly and
ad-dft SCF (via dashboard_worker subprocess), and writes scaling.json/md.

Usage:
    python bench/scaling.py --sizes 2,8,16,32 --ecutwfc 20
    python bench/scaling.py --sizes 2,8,16,32,64,96 --ecutwfc 20 --abacus-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "bench" / "results"
CASE_SRC = ROOT / "tests" / "regress" / "si-gamma"
ABACUS = Path("/root/abacus-develop/build_ci/abacus_basic_para")
PP_DIR = Path("/root/abacus-develop/tests/PP_ORB")
WORKER = ROOT / "bench" / "dashboard_worker.py"


def _base():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import addft as ad
    s = ad.setup_from_case(CASE_SRC)
    return s.geo.lattice, s.geo.frac


def make_supercell(n_atoms: int, ecutwfc: float) -> tuple[Path, tempfile.TemporaryDirectory]:
    """Closest repeat (r1,r2,r3) with 2*r1*r2*r3 == n_atoms."""
    assert n_atoms % 2 == 0, "Si supercell needs even atom count"
    reps = [r for r in range(1, 12) if n_atoms % (2 * r) == 0]
    # Prefer balanced cell, then smallest max repeat.
    best = min(reps, key=lambda r: abs(r - (n_atoms / 2) ** (1 / 3)))
    r1 = best
    rem = n_atoms // (2 * r1)
    r2 = min(r for r in range(1, 12) if rem % r == 0 and
             abs(r - (rem) ** 0.5) == min(abs(x - (rem) ** 0.5)
                                         for x in range(1, 12) if rem % x == 0))
    r3 = rem // r2
    lat0, frac0 = _base()
    lat = np.asarray(lat0) @ np.diag([r1, r2, r3])
    reps = np.array([r1, r2, r3], dtype=float)
    fracs = []
    for i in range(r1):
        for j in range(r2):
            for k in range(r3):
                fracs.append((np.asarray(frac0) + np.array([i, j, k])) / reps)
    fracs = np.concatenate(fracs, axis=0)

    tmp = tempfile.TemporaryDirectory(prefix=f"addft-scale-{n_atoms}-")
    d = Path(tmp.name)
    lines = [
        "ATOMIC_SPECIES", "Si 1.000 Si.pz-vbc.UPF", "",
        "LATTICE_CONSTANT", "1.0", "", "LATTICE_VECTORS",
    ]
    for row in lat:
        lines.append(" ".join(f"{x:.12f}" for x in row))
    lines += ["", "ATOMIC_POSITIONS", "Direct", "Si", "0.0", str(len(fracs))]
    for f in fracs:
        lines.append(" ".join(f"{x:.12f}" for x in f) + " 0 0 0")
    (d / "STRU").write_text("\n".join(lines) + "\n")
    (d / "KPT").write_text("K_POINTS\n0\nGamma\n1 1 1 0 0 0\n")
    inp = f"""INPUT_PARAMETERS
suffix            scale
calculation       scf
ntype             1
nbands            {2 * n_atoms + 2}
symmetry          0
pseudo_dir        {PP_DIR}
ecutwfc           {ecutwfc:g}
scf_thr           1e-8
scf_nmax          100
basis_type        pw
ks_solver         dav_subspace
smearing_method   fixed
cal_force         0
cal_stress        0
"""
    (d / "INPUT").write_text(inp)
    return d, tmp


def run_abacus(case: Path, timeout: int) -> float | None:
    if not ABACUS.exists():
        return None
    shutil.rmtree(case / "OUT.scale", ignore_errors=True)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    t = time.perf_counter()
    try:
        subprocess.run([str(ABACUS)], cwd=case, env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout)
    except Exception:
        return None
    return time.perf_counter() - t


def run_addft(case: Path, timeout: int) -> dict | None:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}:{ROOT}",
        "ADDFT_NATIVE_XC": "1", "ADDFT_NATIVE_EIGH": "1",
        "ADDFT_NATIVE_PACK": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.7",
    })
    cmd = [sys.executable, str(WORKER), "--case", str(case),
           "--scheme", "device_broyden", "--mode", "scf",
           "--driver", "auto", "--repeats", "1", "--scf-only"]
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=timeout)
    except Exception:
        return None
    if p.returncode != 0:
        return {"error": p.stderr[-1000:]}
    line = [x for x in p.stdout.strip().splitlines() if x.startswith("{")][-1]
    return json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2,8,16,32")
    ap.add_argument("--ecutwfc", type=float, default=20.0)
    ap.add_argument("--abacus-timeout", type=int, default=1200)
    ap.add_argument("--addft-timeout", type=int, default=1800)
    ap.add_argument("--abacus-only", action="store_true")
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in sizes:
        case, tmp = make_supercell(n, args.ecutwfc)
        try:
            print(f"[scale] {n} atoms: ABACUS", flush=True)
            t_ab = run_abacus(case, args.abacus_timeout)
            print(f"[scale] {n} atoms: ad-dft", flush=True)
            add = None if args.abacus_only else run_addft(case, args.addft_timeout)
            row = {"n_atoms": n, "ecutwfc": args.ecutwfc,
                   "t_abacus": t_ab,
                   "nG": add.get("nG") if add else None,
                   "nfft": add.get("nfft") if add else None,
                   "n_iter": add.get("n_iter") if add else None,
                   "t_addft": add.get("t_scf") if add and "t_scf" in add else None,
                   "addft_error": add.get("error") if add else None}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            tmp.cleanup()
    (RESULT_DIR / f"scaling-{int(args.ecutwfc)}Ry.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))
    lines = [f"# Scaling: Si, ecutwfc={args.ecutwfc:g} Ry", "",
             "| atoms | nG | nfft | iters | ABACUS 1-core | ad-dft SCF |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['n_atoms']} | {r['nG'] or '—'} | {r['nfft'] or '—'} | "
                     f"{r['n_iter'] or '—'} | {r['t_abacus']:.3g} s | "
                     f"{r['t_addft']:.3g} s |" if r['t_abacus'] and r['t_addft']
                     else f"| {r['n_atoms']} | {r['nG'] or '—'} | {r['nfft'] or '—'} | "
                          f"{r['n_iter'] or '—'} | "
                          f"{r['t_abacus']:.3g} s | skip |"
                     if r['t_abacus'] else f"| {r['n_atoms']} | — | — | — | skip | skip |")
    (RESULT_DIR / f"scaling-{int(args.ecutwfc)}Ry.md").write_text("\n".join(lines) + "\n")
    print(f"[scale] wrote {RESULT_DIR}")


if __name__ == "__main__":
    main()
