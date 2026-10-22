"""L2 原语 · 局域赝势（结构因子随几何重建）(spec/pseudopotential.md §local).

V_loc(G) = (1/Ω) Σ_I ṽ_{s(I)}(|G|) · e^{-iG·R_I}

关键点：结构因子 e^{-iG·R_I} 与 1/Ω 都在图里，因此
  ∂E/∂R_I  → 局域力的 Hellmann–Feynman 项；
  ∂E/∂ε    → 局域势的体积项（−1/Ω 与 G 的缩放）。
两者都不需要单独实现。
"""

from __future__ import annotations

import jax.numpy as jnp

from ..geometry.basis import Basis, g2, gvectors, real_from_g
from ..geometry.cell import Cell, positions, volume
from ..geometry.psp import Pseudo, vloc_q


def local_potential_g(psp: Pseudo, spec_idx, basis: Basis, cell: Cell):
    """(nG,) complex，已是能量的自然单位 Ha（含 1/Ω）。"""
    g = gvectors(basis, cell.lattice)
    q = jnp.sqrt(g2(basis, cell.lattice))
    amp = vloc_q(psp, spec_idx, q)  # (nat, nG)
    r = positions(cell)  # (nat, 3)
    phase = jnp.exp(-1j * (g @ r.T))  # (nG, nat)
    sf = jnp.sum(amp.T * phase, axis=-1)  # (nG,)
    return sf / volume(cell.lattice)


def local_potential_r(psp: Pseudo, spec_idx, basis: Basis, cell: Cell):
    vg = local_potential_g(psp, spec_idx, basis, cell)
    return real_from_g(basis, cell.lattice, vg)


def local_energy(psp: Pseudo, spec_idx, basis: Basis, cell: Cell, rho_r):
    """E_loc = ∫ v_loc(r) n(r) dr，实空间网格积分（可微、无额外约定）。"""
    v = local_potential_r(psp, spec_idx, basis, cell)
    omega = volume(cell.lattice)
    return (omega / basis.nfft) * jnp.sum(v * rho_r), v
