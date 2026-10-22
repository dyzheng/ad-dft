"""契约层测试：把 spec/ 的条文变成断言（CI 的 contract-check）。"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import addft as ad
from addft.geometry.basis import (
    fft_shape,
    is_smooth235,
    make_basis,
    shell_margin,
    smooth_up,
)
from addft.geometry.kmesh import check_weights, gamma, monkhorst_pack
from addft.runtime import units


def test_unit_bridge():
    """ABACUS 是 Ry，内部是 Ha；桥只有一处。"""
    assert units.RY_TO_HA == 0.5
    assert abs(units.HA_TO_EV - 27.211386245988) < 1e-9
    assert abs(units.ry_to_ha(30.0) - 15.0) < 1e-12


def test_smooth_numbers_only_235():
    """ABACUS pw_init.cpp 把 7 的分支注释掉了，因此 7 不允许。"""
    assert is_smooth235(12) and is_smooth235(15) and is_smooth235(27)
    assert not is_smooth235(7) and not is_smooth235(14) and not is_smooth235(49)
    assert smooth_up(7) == 8      # 7 → 8（2³）
    assert smooth_up(11) == 12    # 11 → 12（2²·3）
    assert smooth_up(13) == 15    # 13 → 15（3·5）


def test_fft_box_is_odd_then_smooth():
    """先取 2n+1，再向上取 2·3·5 光滑数。"""
    shape = fft_shape(jnp.eye(3) * 6.0, 2.0, density_factor=4.0)
    assert all(is_smooth235(n) for n in shape)
    assert shape == tuple(shape)


def test_shell_is_frozen_and_margin_reported():
    basis = make_basis(jnp.eye(3) * 6.0, 2.0)
    assert basis.miller.dtype == jnp.int32
    # G=0 必须在
    assert bool(jnp.any(jnp.all(basis.miller == 0, axis=1)))
    m = shell_margin(basis, jnp.eye(3) * 6.0)
    assert 0.0 < m < 1.0


def test_kmers_weights_sum_to_one():
    for km in (gamma(), monkhorst_pack((2, 2, 2)), monkhorst_pack((3, 1, 1), (0.5, 0, 0))):
        assert check_weights(km) < 1e-14


def test_energy_decomposition_tags_are_switchable():
    """不能单独关掉的项，就不能被测量。"""
    from conftest import make_system

    s = make_system()
    st = ad.scf(s)
    full = ad.energy_decomposition(s, st)
    only_kin = ad.energy_decomposition(s, st, active=("kinetic",))
    assert set(full) == {"kinetic", "local", "hartree", "exchange",
                         "correlation", "nonlocal", "ewald"}
    assert set(only_kin) == {"kinetic"}
    assert abs(float(only_kin["kinetic"]) - float(full["kinetic"])) < 1e-12
