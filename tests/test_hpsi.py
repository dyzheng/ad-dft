"""Matrix-free Hψ vs dense Hamiltonian."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import addft as ad
from addft.ops.hpsi import hpsi
from addft.solver.geometry_cache import build_geometry_cache
from addft.solver.scf import effective_potential, hamiltonian
from conftest import make_system


def test_hpsi_matches_dense_hamiltonian():
    s = make_system()
    geo, static = s.geo, s.static
    from addft.solver.scf import initial_density

    rho = initial_density(static, geo)
    cache = build_geometry_cache(geo, static)
    v_r, _ = effective_potential(geo, static, rho, cache=cache)
    H = hamiltonian(geo, static, v_r, cache=cache)

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    psi = (jax.random.normal(key1, (1, static.nband, static.basis.nG))
           + 1j * jax.random.normal(key2, (1, static.nband, static.basis.nG)))
    ref = jnp.einsum("kij,kbj->kbi", H, psi)
    out = hpsi(geo, static, psi, v_r, cache)
    assert float(jnp.max(jnp.abs(out - ref))) < 1e-10
    out_nc = hpsi(geo, static, psi, v_r, cache=None)
    assert float(jnp.max(jnp.abs(out_nc - ref))) < 1e-10


def test_hpsi_multiple_bands():
    s = make_system()
    geo, static = s.geo, s.static
    from addft.solver.scf import initial_density

    rho = initial_density(static, geo)
    cache = build_geometry_cache(geo, static)
    v_r, _ = effective_potential(geo, static, rho, cache=cache)
    H = hamiltonian(geo, static, v_r, cache=cache)

    key1, key2 = jax.random.split(jax.random.PRNGKey(1))
    psi = (jax.random.normal(key1, (1, 3, static.basis.nG))
           + 1j * jax.random.normal(key2, (1, 3, static.basis.nG)))
    ref = jnp.einsum("kij,kbj->kbi", H, psi)
    out = hpsi(geo, static, psi, v_r, cache)
    assert float(jnp.max(jnp.abs(out - ref))) < 1e-10
