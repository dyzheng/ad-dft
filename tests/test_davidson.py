"""Matrix-free Davidson vs dense eigh."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from addft.geometry.basis import gvectors
from addft.ops.hpsi import hpsi
from addft.solver.davidson import davidson_lowest
from addft.solver.geometry_cache import build_geometry_cache
from addft.solver.scf import effective_potential, initial_density
from conftest import make_system


def test_davidson_matches_dense_lowest():
    s = make_system()
    geo, static = s.geo, s.static
    b = static.basis
    rho = initial_density(static, geo)
    cache = build_geometry_cache(geo, static)
    v_r, _ = effective_potential(geo, static, rho, cache=cache)

    from addft.solver.scf import eigh_lowest, hamiltonian
    H = hamiltonian(geo, static, v_r, cache=cache)
    w_ref, v_ref = eigh_lowest(H, static.nband)

    Td = jnp.stack([
        0.5 * jnp.sum(gvectors(b, geo.lattice, static.kpts[ik]) ** 2, axis=-1)
        for ik in range(int(static.kpts.shape[0]))
    ])

    def apply(p):
        return hpsi(geo, static, p.swapaxes(-1, -2), v_r, cache).swapaxes(-1, -2)

    X0 = jnp.broadcast_to(
        jnp.eye(b.nG, dtype=jnp.complex128)[:, :static.nband][None],
        (int(static.kpts.shape[0]), b.nG, static.nband))
    w, V, res = jax.jit(
        lambda x: davidson_lowest(apply, x, static.nband, Td,
                                  max_sub=24, max_iter=30))(X0)
    jax.block_until_ready((w, V, res))
    assert float(jnp.max(jnp.abs(w - w_ref))) < 1e-8
    assert float(res) < 1e-7
    # subspace invariant check
    P = jnp.einsum("kbi,kci->kbc", V, jnp.conj(V))
    Pr = jnp.einsum("kbi,kci->kbc", v_ref, jnp.conj(v_ref))
    assert float(jnp.max(jnp.abs(P - Pr))) < 1e-7
