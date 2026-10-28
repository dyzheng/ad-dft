"""Native CUDA XC kernel gate.

Skipped unless ``native/libaddft_kernels.so`` has been built and a GPU is
available. Run with:

    NVCC=/path/to/nvcc ./native/build.sh
    ADDFT_NATIVE_XC=1 python -m pytest tests/test_native_xc.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from addft.physics import xc

pytestmark = pytest.mark.gpu

if xc._NATIVE_XC is None:
    pytest.skip("native XC kernel not built/available", allow_module_level=True)


def _rho(n=4096):
    key = jax.random.PRNGKey(7)
    return jax.random.uniform(key, (n,), minval=1e-6, maxval=2.0)


def test_native_xc_matches_pointwise_oracle():
    rho = _rho()
    vxc, exc = xc._NATIVE_XC(rho)
    v_ref = jax.vmap(xc.xc_potential_point)(rho)
    e_ref = jnp.sum(xc.lda_energy_density(rho))
    assert float(jnp.max(jnp.abs(vxc - v_ref))) < 1e-12
    assert float(jnp.abs(exc - e_ref) / jnp.abs(e_ref)) < 1e-12


def test_native_vxc_vjp_matches_fxc():
    rho = _rho()
    w = jnp.sin(jnp.arange(rho.size, dtype=rho.dtype) * 0.37)

    def loss(r):
        return jnp.sum(xc.xc_potential_r(xc.xc_energy_lda, None, None, r) * w)

    g = jax.grad(loss)(rho)
    g_ref = xc.xc_kernel_r(rho) * w
    assert float(jnp.max(jnp.abs(g - g_ref))) < 1e-10
