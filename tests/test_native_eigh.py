"""Native cuSOLVER eigensolver gate."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from addft.solver.eigh import _NATIVE_EIGH, eigh_lowest

pytestmark = pytest.mark.gpu

if _NATIVE_EIGH is None:
    pytest.skip("native eigensolver not built/available", allow_module_level=True)


def _hermitian(nk=2, n=96):
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    A = (jax.random.normal(k1, (nk, n, n)) +
         1j * jax.random.normal(k2, (nk, n, n)))
    return 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))


@pytest.mark.parametrize("nband", [1, 4, 8])
def test_native_eigh_matches_jax(nband):
    H = _hermitian()
    w, v = eigh_lowest(H, nband)
    w_ref, v_ref = jnp.linalg.eigh(H)
    w_ref = w_ref[:, :nband]
    v_ref = jnp.swapaxes(v_ref[:, :, :nband], -1, -2)

    assert float(jnp.max(jnp.abs(w - w_ref))) < 1e-10
    # subspace projector is the right invariant under degeneracy/phase
    P = jnp.einsum("kbi,kci->kbc", v, jnp.conj(v))
    P_ref = jnp.einsum("kbi,kci->kbc", v_ref, jnp.conj(v_ref))
    assert float(jnp.max(jnp.abs(P - P_ref))) < 1e-10

    residual = jnp.einsum("kij,kbj->kbi", H, v) - v * w[:, :, None]
    assert float(jnp.max(jnp.abs(residual))) < 1e-10
