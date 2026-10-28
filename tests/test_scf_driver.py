"""Device SCF driver vs Python/DIIS reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import addft as ad
from addft.physics import xc
from conftest import make_system

pytestmark = pytest.mark.gpu

if not hasattr(xc, "_NATIVE_XC") or (xc.USE_LIBXC and xc._NATIVE_XC is None):
    pytest.skip("device SCF needs native XC or ADDFT_XC=analytic",
                allow_module_level=True)


def test_device_scf_matches_python_driver():
    s = make_system()
    st_dev = ad.scf(s, driver="device")
    st_py = ad.scf(s, driver="python")

    assert float(jnp.max(jnp.abs(st_dev.rho_r - st_py.rho_r))) < 1e-10
    assert float(jnp.max(jnp.abs(st_dev.eig - st_py.eig))) < 1e-9
    assert float(jnp.max(jnp.abs(st_dev.occ - st_py.occ))) < 1e-9
    assert float(st_dev.residual) < 1e-9


def test_device_scf_energy_force_stress_match_python():
    s = make_system()
    E_d, F_d = ad.energy_and_forces(s, driver="device")
    E_p, F_p = ad.energy_and_forces(s, driver="python")
    S_d = ad.stress(s, driver="device")
    S_p = ad.stress(s, driver="python")

    assert abs(float(E_d - E_p)) < 1e-10
    assert float(jnp.max(jnp.abs(F_d - F_p))) < 1e-10
    assert float(jnp.max(jnp.abs(S_d - S_p))) < 1e-10
