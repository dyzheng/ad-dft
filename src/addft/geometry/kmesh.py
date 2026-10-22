"""L1 k 网格 (spec/invariants.md §kweight).

权重契约（最容易出错、也最难发现的一类）：Σ_k w_k = 1，**不是** 1/nk 再乘自旋因子。
自旋简并因子 2 在密度与能量里单独出现，绝不混进 w_k。
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class KMesh(NamedTuple):
    kpts: jnp.ndarray  # (nk, 3) 笛卡尔，单位 2π/Bohr
    weights: jnp.ndarray  # (nk,)  和为 1


def gamma() -> KMesh:
    return KMesh(
        kpts=jnp.zeros((1, 3)),
        weights=jnp.ones((1,)),
    )


def monkhorst_pack(n, shift=None) -> KMesh:
    """n = (n1,n2,n3) 网格；shift 为 0/0.5 平移（与 ABACUS KPT 文件语义一致）。"""
    n = tuple(int(x) for x in n)
    shift = (0.0, 0.0, 0.0) if shift is None else tuple(float(x) for x in shift)
    pts = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                pts.append(
                    (
                        (i + shift[0]) / n[0],
                        (j + shift[1]) / n[1],
                        (k + shift[2]) / n[2],
                    )
                )
    frac = np.asarray(pts, dtype=float)
    frac = np.where(frac >= 0.5, frac - 1.0, frac)  # 折回 [-0.5, 0.5)
    nk = frac.shape[0]
    # 注意：这里 (…)/n 给出的是「以 2π/bohr 为单位」的 k，与 reciprocal() 同量纲
    return KMesh(
        kpts=2.0 * jnp.pi * jnp.asarray(frac),
        weights=jnp.full((nk,), 1.0 / nk),
    )


def check_weights(kmesh: KMesh, tol: float = 1e-14) -> float:
    return float(jnp.abs(jnp.sum(kmesh.weights) - 1.0))
