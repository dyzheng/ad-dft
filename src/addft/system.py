"""系统静态数据 + 可微几何的分离（方案附录 D 设计要点一）。

`geo`（Cell）是**唯一**的可微叶子；`static` 里的东西一律不参与微分：
  * 冻结的整数指标 miller
  * k 网格与权重
  * 赝势数值（zval / D / 径向参数）
  * nband / 自旋简并 / σ
这不是洁癖：把 static 拖进图里会让显存爆炸，并且产生没有物理意义的导数
（∂E/∂nband 是什么？没人想知道，但 AD 会算给你）。
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .geometry.basis import Basis, make_basis
from .geometry.cell import Cell
from .geometry.kmesh import KMesh
from .geometry.psp import Pseudo


class Static(NamedTuple):
    basis: Basis
    kpts: jnp.ndarray
    kweights: jnp.ndarray
    psp: Pseudo
    spec_idx: jnp.ndarray
    nband: int
    spin_degeneracy: float
    sigma: float
    nelec: float
    eta_ewald: float


class SCFState(NamedTuple):
    psi: jnp.ndarray  # (nk, nband, nG) complex
    eig: jnp.ndarray  # (nk, nband)
    occ: jnp.ndarray  # (nk, nband)
    mu: jnp.ndarray  # ()
    rho_r: jnp.ndarray  # (nfft,)
    n_iter: jnp.ndarray
    residual: jnp.ndarray


def make_static(cell: Cell, psp: Pseudo, kmesh: KMesh, ecut: float,
                nband: int, spec_idx=None, sigma: float = 0.01,
                density_factor: float = 4.0, eta_ewald: float = 0.9) -> Static:
    if spec_idx is None:
        spec_idx = jnp.zeros((cell.natom,), dtype=jnp.int32)
    spec_idx = jnp.asarray(spec_idx, dtype=jnp.int32)
    if int(psp.nspec) != 1:
        raise NotImplementedError("多 species 在 R3 只允许显式拒绝，不做静默映射")
    kpts = kmesh.cartesian(cell.lattice)
    basis = make_basis(cell.lattice, ecut, density_factor, kpts=kpts)
    nelec = float(jnp.sum(psp.zval[spec_idx]))
    return Static(
        basis=basis,
        kpts=kpts,
        kweights=kmesh.weights,
        psp=psp,
        spec_idx=spec_idx,
        nband=int(nband),
        spin_degeneracy=2.0,
        sigma=float(sigma),
        nelec=nelec,
        eta_ewald=float(eta_ewald),
    )
