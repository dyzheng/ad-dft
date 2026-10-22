import jax.numpy as jnp
import pytest

import addft as ad
from addft.geometry.basis import make_basis
from addft.geometry.kmesh import check_weights, monkhorst_pack
from addft.solver.scf import kinetic_diag


def test_mp_2x2x2_is_in_first_bz():
    lat = jnp.eye(3) * 6.0
    km = monkhorst_pack((2, 2, 2))
    k = km.cartesian(lat)
    assert k.shape == (8, 3)
    # 6 Bohr 立方晶胞：b = 2π/6；2×2×2 的 k 分量只能是 0 或 -π/6
    assert float(jnp.max(jnp.abs(k))) == pytest.approx(jnp.pi / 6.0, abs=1e-12)
    assert check_weights(km) < 1e-14


def test_free_electron_kinetic_at_shifted_k():
    lat = jnp.eye(3) * 6.0
    basis = make_basis(lat, 2.0)
    km = monkhorst_pack((2, 2, 2))
    k = km.cartesian(lat)[1]
    t = kinetic_diag(basis, lat, k)
    # G=0 的动能 = |k|²/2
    assert float(t[0]) == pytest.approx(float(jnp.sum(k * k) / 2.0), abs=1e-12)


def test_non_gamma_scf_is_self_consistent():
    s = ad.setup(lattice=jnp.eye(3) * 6.0,
                 frac=jnp.array([[0.0, 0.0, 0.0], [0.53, 0.47, 0.49]]),
                 kmesh=monkhorst_pack((2, 2, 2)),
                 ecut=2.0, nband=2)
    st = ad.scf(s)
    assert float(st.residual) < 1e-8
    assert jnp.isfinite(ad.energy(s, st))
