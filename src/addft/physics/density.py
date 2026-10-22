"""L2 原语 · 密度合成。

n(r) = Σ_nk w_k f_nk · g_s · |ψ_nk(r)|^2

三个高频错误（spec/invariants.md 已写成测试）：
  * w_k 归一化成 1/nk 而不是 1（能量差一个 nk 因子，但力仍然对——最难发现的一类）；
  * 自旋简并因子 g_s 漏乘或重乘；
  * 直接对 ψ_nk 与 ψ_nk' 相加后再取模（k 点之间相位不可比，必须各自取模方）。
"""

from __future__ import annotations

import jax.numpy as jnp

from ..geometry.basis import Basis, psi_to_real


def density_from_psi(basis: Basis, lattice, psi, occ, weights, spin_degeneracy=2.0):
    """psi: (nk, nband, nG) complex; occ: (nk, nband); weights: (nk,) → (nfft,)."""
    psi_r = psi_to_real(basis, lattice, psi)  # (nk, nband, nfft)
    mod2 = jnp.real(psi_r * jnp.conj(psi_r))
    w = weights[:, None] * occ * spin_degeneracy  # (nk, nband)
    return jnp.einsum("kb,kbi->i", w, mod2)


def nelec_from_occ(occ, weights, spin_degeneracy=2.0):
    return jnp.sum(weights[:, None] * occ) * spin_degeneracy
