"""注入 3 类已知 bug，确认 oracle 真的会失败（不是只会打印）。

注入后仍通过，说明该 oracle 不够强；这里用能直接抓住该类错误的判据：
  * XC 势权重错 -> 点级 v_xc 与 d(nε)/dn 的 oracle；
  * D 布局错  -> 直接 V_NL Hermitian/块位置判据；
  * k 单位错  -> 第一 BZ 的 |k| 上限判据（自由电子自洽式检查抓不住任何 k）。
"""

import jax.numpy as jnp
import pytest

from test_kpoints import test_mp_2x2x2_is_in_first_bz
from test_primitives import (
    test_nonlocal_operator_is_hermitian,
    test_xc_potential_matches_pointwise_oracle,
)


def test_harness_catches_wrong_xc_weight(monkeypatch, system):
    import addft.physics.xc as xc

    orig = xc.xc_potential_r
    monkeypatch.setattr(xc, "xc_potential_r",
                        lambda *a, **k: 0.5 * orig(*a, **k))
    with pytest.raises(AssertionError):
        test_xc_potential_matches_pointwise_oracle(system)


def test_harness_catches_bad_dion(monkeypatch, system):
    import addft.physics.nonlocal_op as nl

    good_fn = nl.dion_full

    def old_layout(psp, spec_idx):
        good = good_fn(psp, spec_idx)
        nat, nh = int(spec_idx.shape[0]), int(psp.nh[int(spec_idx[0])])
        d = jnp.zeros((nat, nat, nh, nh), dtype=good.dtype)
        for i in range(nat):
            d = d.at[i, i].set(good[i * nh:(i + 1) * nh,
                                     i * nh:(i + 1) * nh])
        return d.reshape(nat * nh, nat * nh)

    monkeypatch.setattr(nl, "dion_full", old_layout)
    with pytest.raises(AssertionError):
        test_nonlocal_operator_is_hermitian(system)


def test_harness_catches_bad_kpoint_unit(monkeypatch):
    from addft.geometry.kmesh import KMesh

    monkeypatch.setattr(
        KMesh, "cartesian",
        lambda self, lattice: 2.0 * jnp.pi * self.kfrac)
    with pytest.raises(AssertionError):
        test_mp_2x2x2_is_in_first_bz()
