"""L1 几何 · Cell 是 pytree 叶子 (spec/discretization.md §cell).

本文件只做两件事：把晶格/分数坐标变成可微张量，并提供应变算子。
**不在这里做任何「摊平」**——一旦 Cell 被转成 numpy 数组，L2 之后所有导数都断了，
而能量仍然是对的（这是项目早期最贵的一类错误，见 APPENDIX 失败模式 F-1）。
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

TWO_PI = 2.0 * jnp.pi


class Cell(NamedTuple):
    """晶胞。lattice 行向量为 a1,a2,a3（Bohr）；frac 为分数坐标。"""

    lattice: jnp.ndarray  # (3, 3)
    frac: jnp.ndarray  # (nat, 3)

    @property
    def natom(self) -> int:
        return int(self.frac.shape[0])


def reciprocal(lattice):
    """倒格矢矩阵，行向量 b1,b2,b3，满足 a_i · b_j = 2π δ_ij。"""
    return TWO_PI * jnp.linalg.inv(lattice).T


def volume(lattice):
    return jnp.abs(jnp.linalg.det(lattice))


def positions(cell: Cell):
    """笛卡尔坐标 (nat, 3)，可微（对 frac 与 lattice 双向）。"""
    return cell.frac @ cell.lattice


def with_frac(cell: Cell, frac) -> Cell:
    return Cell(lattice=cell.lattice, frac=frac)


def strained(cell: Cell, eps) -> Cell:
    """施加均匀应变：a_i' = a_i (I + ε)，分数坐标固定。

    这是应力定义的离散化载体：σ_αβ = (1/Ω) dE/dε_αβ。
    因为 G 由 lattice 通过 reciprocal() 算出，壳层跳变被「冻结整数指标」吸收。
    """
    eye = jnp.eye(3, dtype=cell.lattice.dtype)
    return Cell(lattice=cell.lattice @ (eye + eps), frac=cell.frac)


def displacement(cell: Cell, delta) -> Cell:
    """按笛卡尔位移移动原子：Δ frac = delta @ inv(lattice)。"""
    dfrac = delta @ jnp.linalg.inv(cell.lattice)
    return with_frac(cell, cell.frac + dfrac)
