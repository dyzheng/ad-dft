"""Native complex128 scatter/gather gate."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from addft.geometry import basis as B
from conftest import make_system

pytestmark = pytest.mark.gpu

if B._NATIVE_PACK is None:
    pytest.skip("native pack kernels not built/available", allow_module_level=True)


def test_pack_scatter_gather_matches_jax():
    s = make_system()
    b = s.static.basis
    idx = B._flat_index(b.fft_shape, b.miller)
    c = (jnp.sin(jnp.arange(b.nG) * 0.37)
         + 1j * jnp.cos(jnp.arange(b.nG) * 0.53))

    g = B.coeffs_to_grid(b, c)
    flat = jnp.zeros((b.nfft,), dtype=c.dtype)
    g_ref = flat.at[idx].set(c, unique_indices=True).reshape(b.fft_shape)
    assert float(jnp.max(jnp.abs(g - g_ref))) == 0.0

    cc = B.grid_to_coeffs(b, g)
    cc_ref = jnp.take(g.reshape(-1), idx)
    assert float(jnp.max(jnp.abs(cc - cc_ref))) == 0.0


def test_pack_batched_scatter():
    s = make_system()
    b = s.static.basis
    idx = B._flat_index(b.fft_shape, b.miller)
    c = (jnp.sin(jnp.arange(2 * 3 * b.nG) * 0.11).reshape(2, 3, b.nG)
         + 1j * jnp.cos(jnp.arange(2 * 3 * b.nG) * 0.07).reshape(2, 3, b.nG))

    g = B.coeffs_to_grid(b, c)
    flat = jnp.zeros((b.nfft,), dtype=c.dtype)
    one = flat.at[idx].set(c[0, 0], unique_indices=True).reshape(b.fft_shape)
    assert float(jnp.max(jnp.abs(g[0, 0] - one))) == 0.0
    assert g.shape == (2, 3) + tuple(b.fft_shape)


def test_pack_c64_matches_jax():
    import addft.kernels.pack as P
    if not P._C64_READY:
        pytest.skip("native complex64 pack not available")
    s = make_system()
    b = s.static.basis
    idx = B._flat_index(b.fft_shape, b.miller)
    c = (jnp.sin(jnp.arange(b.nG) * 0.17)
         + 1j * jnp.cos(jnp.arange(b.nG) * 0.23)).astype(jnp.complex64)
    g = B.coeffs_to_grid(b, c)
    flat = jnp.zeros((b.nfft,), dtype=c.dtype)
    g_ref = flat.at[idx].set(c, unique_indices=True).reshape(b.fft_shape)
    assert float(jnp.max(jnp.abs(g - g_ref))) == 0.0
    cc = B.grid_to_coeffs(b, g)
    assert float(jnp.max(jnp.abs(cc - jnp.take(g.reshape(-1), idx)))) == 0.0
