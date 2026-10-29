"""One candidate evaluation for bench/evolve.py."""

from __future__ import annotations

import argparse
import json
import time

import jax
import addft as ad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--energy", action="store_true")
    args = p.parse_args()
    s = ad.setup_from_case(args.case)
    # Warm up (compile) before timing; otherwise every candidate is dominated
    # by the same one-time XLA/NVRTC/cuSOLVER cost.
    st = ad.scf(s, eigh_solver="auto")
    jax.block_until_ready(st.rho_r)
    t = time.perf_counter()
    st = ad.scf(s, eigh_solver="auto")
    jax.block_until_ready(st.rho_r)
    t_scf = time.perf_counter() - t
    out = dict(
        nG=int(s.static.basis.nG), nfft=int(s.static.basis.nfft),
        n_iter=int(st.n_iter), residual=float(st.residual),
        t_scf=t_scf,
    )
    if args.energy:
        out["E"] = float(ad.energy(s, st))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
