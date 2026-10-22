"""L5 观测量 · 约束（DeltaQ / DeltaSpin 的数学骨架）。

关键区别，也是本项目"何时需要响应项"的分水岭：

* **能量** `E(geo)`：收敛点对 ψ 驻定（包络定理）⇒ 一阶导只需状态冻结，
  `λ = 0`。见 `api.make_energy(mode="envelope")`。
* **非能量泛函** `C(w) = ∫ w(r) n(r) dr`：它不是变分泛函，`λ ≠ 0`，
  必须解 `(I − ∂F/∂n)ᵀλ = ∂C/∂n`。这就是 DeltaQ / DeltaP / Born 电荷的数学结构。

约束 DFT 的拉格朗日结构：对 `C_i[ρ] = C_target,i`，
    ℒ = E − Σ λ_i (C_i − C_target,i)，收敛点 `∂E/∂C_target,i = −λ_i`。
所以"改变约束目标 → 能量如何变"= 一次伴随，而不是每个维度两次 SCF。
对 N 个原子 × 3 个方向，传统做法是 6N 次 SCF；这里是 1 次伴随 + N_c 个标量。
"""

from __future__ import annotations

import jax.numpy as jnp

from ..geometry.cell import Cell, volume
from ..system import SCFState, Static

FOUR_PI = 4.0 * jnp.pi


def delta_q(geo: Cell, static: Static, state: SCFState, weight) -> jnp.ndarray:
    """C = ∫ w(r) n(r) dr = (Ω/N) Σ_j w_j n_j。

    `weight` 必须由契约给定（Hirshfeld / Voronoi / Roby–Sigma 给出不同的
    "原子电荷"，且导数也不同）。权重方案是输入的一部分，不同方案的结果不可混用。
    """
    omega = volume(geo.lattice)
    return (omega / static.basis.nfft) * jnp.sum(weight * state.rho_r)


def delta_q_grad(geo: Cell, static: Static, state: SCFState, weight):
    """dC/dθ —— 必须走伴随（C 不是驻点泛函）。

    返回 (grad_cell, diagnostics)。
    """
    from ..diff.implicit import implicit_grad

    loss = lambda g_, s_, st_: delta_q(g_, s_, st_, weight)
    return implicit_grad(geo, static, state, loss, return_diagnostics=True)


def make_gaussian_weight(basis, center, sigma=0.8, lattice=None):
    """构造一个归一化权重（用于测试与 ΔQ 原型）。"""
    shape = tuple(basis.fft_shape)
    idx = jnp.stack(jnp.meshgrid(*(jnp.arange(n) for n in shape), indexing="ij"), axis=-1)
    frac = idx.reshape(-1, 3) / jnp.array(shape, dtype=jnp.float64)
    r = frac @ lattice
    d2 = jnp.sum((r - center) ** 2, axis=-1)
    w = jnp.exp(-0.5 * d2 / sigma ** 2)
    return w / jnp.sum(w) * basis.nfft  # 使 (Ω/N)Σw 的量级为 1
