"""Si/Γ L3 对照（需要本地 ABACUS 与 MPI）。"""

import os

import jax.numpy as jnp
import pytest

import addft as ad
from addft.runtime.units import HA_TO_EV, HA_TO_KBAR
from adapters.abacus.runner import run_case

pytestmark = [
    pytest.mark.slow,
    pytest.mark.abacus,
    pytest.mark.skipif(os.environ.get("ADDFT_RUN_ABACUS") != "1",
                       reason="设置 ADDFT_RUN_ABACUS=1 才运行 ABACUS 对照"),
]


def test_si_gamma_matches_abacus():
    ref = run_case("tests/regress/si-gamma", suffix="r3")
    s = ad.setup_from_case("tests/regress/si-gamma")
    st = ad.scf(s, max_iter=100)
    e_ev = float(ad.energy(s, st)) * HA_TO_EV
    assert abs(e_ev - ref["etot_ev"]) / 2.0 < 0.010  # 10 meV/atom

    ref_f = run_case("tests/regress/si-distorted", suffix="r3f")
    sf = ad.setup_from_case("tests/regress/si-distorted")
    f_ev_a = ad.energy_and_forces(sf)[1] * HA_TO_EV / 0.529177210903
    assert float(jnp.max(jnp.abs(f_ev_a - ref_f["forces"]))) < 0.010

    sigma = ad.stress(s) * HA_TO_KBAR
    n = min(sigma.shape[0], ref["stress"].shape[0])
    assert float(jnp.max(jnp.abs(sigma[:n, :n] - ref["stress"][:n, :n]))) < 5.0
