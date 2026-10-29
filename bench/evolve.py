"""Small self-evolution loop for the Davidson/mixer control parameters.

Candidate dimensions are environment-exposed in :mod:`addft.solver.scf`:
  ADDFT_DAVID_MAX_SUB, ADDFT_DAVID_COLD, ADDFT_DAVID_WARM,
  ADDFT_DAVID_PRECOND, ADDFT_BROYDEN_HI, ADDFT_BROYDEN_LO

The objective is SCF wall time with a residual penalty (and optional energy
penalty vs ABACUS). This is deliberately a closed loop: evaluate in fresh
processes, select elites, mutate, deduplicate, record.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "bench" / "results"
WORKER = ROOT / "bench" / "evolve_worker.py"
ABACUS_LOG = Path("/tmp/abacus_si_100/OUT.cmp/running_scf.log")

SPACE = dict(
    max_sub=[12, 18, 24, 30, 36, 48, 60],
    cold=[8, 10, 12, 15, 20, 25],
    warm=[3, 5, 7, 9, 10, 12],
    precond=["abacus", "kinetic"],
    pre_alpha=[0.5, 0.75, 1.0, 1.25, 1.5],
    hi=[0.8, 0.9, 1.0],
    lo=[0.4, 0.5, 0.6],
)


def key(c):
    return (c["max_sub"], c["cold"], c["warm"], c["precond"],
            c.get("pre_alpha", 1.0), c["hi"], c["lo"])


def score(r, ref_e=None):
    s = r.get("t_scf", 1e9)
    res = r.get("residual", 1.0)
    s += 100.0 * max(0.0, res - 1e-9) / 1e-9
    if ref_e is not None and "E" in r:
        s += 1e4 * abs(r["E"] - ref_e)
    return s


def evaluate(case, c, check_energy, ref_e, timeout):
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}:{ROOT}",
        "ADDFT_NATIVE_XC": "1", "ADDFT_NATIVE_EIGH": "1",
        "ADDFT_NATIVE_PACK": "1",
        "ADDFT_DAVID_MAX_SUB": str(c["max_sub"]),
        "ADDFT_DAVID_COLD": str(c["cold"]),
        "ADDFT_DAVID_WARM": str(c["warm"]),
        "ADDFT_DAVID_PRECOND": c["precond"],
        "ADDFT_DAVID_PRECOND_ALPHA": str(c.get("pre_alpha", 1.0)),
        "ADDFT_BROYDEN_HI": str(c["hi"]),
        "ADDFT_BROYDEN_LO": str(c["lo"]),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.7",
    })
    cmd = [sys.executable, str(WORKER), "--case", str(case)]
    if check_energy:
        cmd.append("--energy")
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=timeout)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if p.returncode != 0:
        return {"error": p.stderr[-1500:]}
    line = [x for x in p.stdout.strip().splitlines() if x.startswith("{")][-1]
    return json.loads(line)


def mutate(parents, rng):
    c = dict(rng.choice(parents))
    for _ in range(rng.choice([1, 1, 2])):
        k = rng.choice(list(SPACE))
        vals = [v for v in SPACE[k] if v != c[k]]
        c[k] = rng.choice(vals)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--check-energy", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    tmp = None
    if args.case:
        case = Path(args.case)
    else:
        sys.path.insert(0, str(ROOT))
        import bench.scaling as sc
        case, tmp = sc.make_supercell(2, 100.0)

    ref_e = None
    if ABACUS_LOG.exists():
        try:
            sys.path.insert(0, str(ROOT))
            from adapters.abacus.runner import parse_running_scf
            ref_e = parse_running_scf(ABACUS_LOG)["etot_ha"]
        except Exception:
            ref_e = None

    base = dict(max_sub=48, cold=15, warm=5, precond="abacus", pre_alpha=1.0, hi=0.9, lo=0.5)
    population = [base]
    while len(population) < args.population:
        population.append(mutate([base], rng))
    seen = {}
    history = []
    try:
        for gen in range(args.generations):
            print(f"[evolve] generation {gen + 1}", flush=True)
            for c in population:
                k = key(c)
                if k in seen:
                    continue
                print(f"[evolve] eval {c}", flush=True)
                r = evaluate(case, c, args.check_energy, ref_e, args.timeout)
                seen[k] = r
                history.append({"generation": gen + 1, "candidate": c,
                                "result": r,
                                "score": score(r, ref_e) if "error" not in r else 1e12})
                print(f"[evolve] -> {r}", flush=True)
            ranked = sorted(history, key=lambda x: x["score"])
            elites = []
            for x in ranked:
                if key(x["candidate"]) not in [key(e["candidate"]) for e in elites]:
                    elites.append(x)
                if len(elites) >= args.elite:
                    break
            if gen + 1 == args.generations:
                break
            parents = [e["candidate"] for e in elites]
            population = list(parents)
            while len(population) < args.population:
                population.append(mutate(parents, rng))
    finally:
        if tmp is not None:
            tmp.cleanup()

    best = min(history, key=lambda x: x["score"])
    (RESULT_DIR / "evolution.json").write_text(json.dumps(history, indent=2, ensure_ascii=False))
    lines = ["# Davidson/mixer evolution", "",
             "| gen | max_sub | cold | warm | precond | alpha | hi | lo | SCF | iters | residual | score |",
             "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for h in history:
        c, r = h["candidate"], h["result"]
        if "error" in r:
            lines.append(f"| {h['generation']} | {c['max_sub']} | {c['cold']} | {c['warm']} | "
                         f"{c['precond']} | {c['hi']} | {c['lo']} | ERROR | — | — | — |")
        else:
            lines.append(f"| {h['generation']} | {c['max_sub']} | {c['cold']} | {c['warm']} | "
                         f"{c['precond']} | {c.get('pre_alpha',1.0)} | {c['hi']} | {c['lo']} | {r['t_scf']:.3f} | "
                         f"{r['n_iter']} | {r['residual']:.2e} | {h['score']:.3f} |")
    (RESULT_DIR / "evolution.md").write_text("\n".join(lines) + "\n")
    print("[evolve] best:", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
