"""L3 求解器 · 几何缓存。

SCF 迭代内 ``geo`` 固定，``v_loc``、非局域投影与 ``D`` 只依赖几何，
与密度无关。原实现每个迭代把它们全部重建，实测占一次 ``scf_map`` 的
约 57%（``nG=341``）。这里把它们在 ``solve_scf`` 入口算一次，之后复用。

边界：
  * 缓存只用于**正向 SCF**。对 ``geo`` 求导时不能复用旧几何的缓存；
    ``scf_map`` 的 JVP 路径显式忽略缓存。
  * ``geo`` 改变（结构优化、应力应变）必须重建缓存。调用方负责生命周期。
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from ..geometry.basis import Basis
from ..geometry.cell import Cell
from ..physics import nonlocal_op
from ..physics.local import local_potential_r
from ..system import Static


class GeometryCache(NamedTuple):
    v_loc_r: jnp.ndarray      # (nfft,) 实空间局域势
    proj: jnp.ndarray         # (nk, nproj, nG) 非局域投影
    dion: jnp.ndarray         # (nproj, nproj)


def build_geometry_cache(geo: Cell, static: Static) -> GeometryCache:
    basis: Basis = static.basis
    return GeometryCache(
        v_loc_r=local_potential_r(static.psp, static.spec_idx, basis, geo),
        proj=nonlocal_op.projectors(static.psp, static.spec_idx, basis, geo,
                                    static.kpts),
        dion=nonlocal_op.dion_full(static.psp, static.spec_idx),
    )
