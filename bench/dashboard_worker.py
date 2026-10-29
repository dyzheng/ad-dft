"""Worker for bench/dashboard.py: one case/scheme in a fresh process."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp

import addft as ad


def _med(fn, reps):
    fn()
    jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def scf_main(args):
    s = ad.setup_from_case(args.case)
    st = ad.scf(s, driver=args.driver, max_iter=200,
                eigh_solver=args.eigh_solver)
    jax.block_until_ready(st.rho_r)

    t_scf = _med(lambda: ad.scf(s, driver=args.driver, max_iter=200,
                                eigh_solver=args.eigh_solver), args.repeats)
    out = dict(
        mode="scf", scheme=args.scheme, case=args.case,
        nG=int(s.static.basis.nG), nfft=int(s.static.basis.nfft),
        n_iter=int(st.n_iter), residual=float(st.residual),
        t_scf=t_scf,
    )
    if not args.scf_only:
        e_fn = ad.make_energy(s.static, mode="envelope", driver=args.driver,
                              eigh_solver=args.eigh_solver)
        e = e_fn(s.geo)
        f = ad.energy_and_forces(s, driver=args.driver,
                                 eigh_solver=args.eigh_solver)[1]
        out["t_ef"] = _med(
            lambda: ad.energy_and_forces(s, driver=args.driver,
                                         eigh_solver=args.eigh_solver),
            args.repeats)
        out["E"] = float(e)
        out["F_max"] = float(jnp.max(jnp.abs(f)))
    if args.stress and not args.scf_only:
        sig = ad.stress(s, driver=args.driver)
        out["t_stress"] = _med(lambda: ad.stress(s, driver=args.driver), args.repeats)
        out["stress_max"] = float(jnp.max(jnp.abs(sig)))
    print(json.dumps(out))


def micro_main(args):
    from addft.geometry import basis as B
    from addft.ops.hpsi import hpsi
    from addft.solver.eigh import eigh_lowest
    from addft.solver.geometry_cache import GeometryCache, build_geometry_cache
    from addft.solver.scf import effective_potential, initial_density

    s = ad.setup_from_case(args.case)
    geo, static = s.geo, s.static
    rho = initial_density(static, geo)
    cache = build_geometry_cache(geo, static)
    v_r, _ = effective_potential(geo, static, rho, cache=cache)
    psi64 = (jax.random.normal(jax.random.PRNGKey(0), (1, static.nband, static.basis.nG))
             + 1j * jax.random.normal(jax.random.PRNGKey(1), (1, static.nband, static.basis.nG)))
    cache32 = GeometryCache(v_loc_r=cache.v_loc_r.astype(jnp.float32),
                            proj=cache.proj.astype(jnp.complex64),
                            dion=cache.dion.astype(jnp.complex64))

    # Build one dense H for eigh benchmarking only.
    from addft.solver.scf import hamiltonian
    H = hamiltonian(geo, static, v_r, cache=cache)

    def h64(p):
        return hpsi(geo, static, p, v_r, cache)

    def h32(p):
        return hpsi(geo, static, p.astype(jnp.complex64),
                    v_r.astype(jnp.float32), cache32)

    f_h64 = jax.jit(h64)
    f_h32 = jax.jit(h32)
    f_eigh = jax.jit(lambda h: eigh_lowest(h, static.nband))
    f_dense = jax.jit(lambda h: jnp.linalg.eigh(h))

    t_h64 = _med(lambda: f_h64(psi64), args.repeats)
    t_h32 = _med(lambda: f_h32(psi64), args.repeats)
    t_eigh = _med(lambda: f_eigh(H), max(1, args.repeats // 2))
    t_dense = _med(lambda: f_dense(H), max(1, args.repeats // 2))
    print(json.dumps(dict(
        mode="micro", scheme=args.scheme, case=args.case,
        nG=int(static.basis.nG), nfft=int(static.basis.nfft),
        hpsi_fp64=t_h64 * 1e3, hpsi_fp32=t_h32 * 1e3,
        eigh_native=t_eigh, eigh_dense=t_dense,
    )))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--scheme", required=True)
    p.add_argument("--mode", choices=["scf", "micro"], default="scf")
    p.add_argument("--driver", default="auto")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--stress", action="store_true")
    p.add_argument("--scf-only", action="store_true")
    p.add_argument("--eigh-solver", default="auto",
                   choices=["auto", "dense", "davidson"])
    args = p.parse_args()
    if args.mode == "scf":
        scf_main(args)
    else:
        micro_main(args)
