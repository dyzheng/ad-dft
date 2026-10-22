"""共享夹具：所有测试用同一个最小体系（Γ 点、2 原子、合成赝势）。

参数选择理由：
  lattice=6 Bohr, ecut=2 Ha, density_factor=4
    → nG=27, nfft=729，Nyquist 4.71 > 2|G|max=4.0（满足纯网格表示的判据）
  这套参数让每个测试在 5–20 s 内跑完，同时保留全部数值结构。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import addft as ad  # noqa: E402

LATTICE = jnp.eye(3) * 6.0
# 低对称构型：破坏所有点群对称性（协议第三步：单一构型的 FD 通过不算数）
FRAC_DISTORTED = jnp.array([[0.0, 0.0, 0.0], [0.53, 0.47, 0.49]])
FRAC_SYMMETRIC = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def make_system(frac=FRAC_DISTORTED, ecut=2.0, nband=2, factor=4.0, sigma=0.05):
    return ad.setup(lattice=LATTICE, frac=frac, ecut=ecut, nband=nband,
                    density_factor=factor, sigma=sigma)


@pytest.fixture(scope="session")
def system():
    return make_system()


@pytest.fixture(scope="session")
def state(system):
    return ad.scf(system)

