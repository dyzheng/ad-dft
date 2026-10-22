"""L2 原语 · Ewald 离子-离子能 (spec/units.md §ewald).

E_II = E_real + E_recip + E_self + E_background
  E_real   = 1/2 Σ_{I,J} Σ_{T≠0(当 I=J)} Z_I Z_J erfc(η r)/r
  E_recip  = (2π/Ω) Σ_{G≠0} e^{-G²/4η²}/G² |Σ_I Z_I e^{-iG·R_I}|²
  E_self   = -(η/√π) Σ_I Z_I²
  E_bg     = -(π/(2Ωη²)) (Σ_I Z_I)²      ← 与 Hartree 的 G=0 置零配套

E_background 不能因为「体系中性」就省掉：中性的是「离子 + 电子」，
离子自身的 Σ Z ≠ 0，背景项由电子的 G=0 项补偿，两者必须成对出现。

oracle：η 无关性（扫 η，能量变化 < 1e-8 Ha）、平移不变、AD 对 R 的导数 vs 中心差分。

默认 ``nmax=8`` 不是保守取值而是收敛值：5.1 Bohr 的 Si 原胞在 ``nmax=1``
时倒空间求和误差约 0.069 Ha（≈0.9 eV/atom），会污染与 ABACUS 的能量对照。
``nmax=8`` 对该原胞与 6 Bohr 合成盒均与高截断参考一致到 1e-8 Ha。
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp

from ..geometry.cell import Cell, positions, volume

DEFAULT_ETA = 0.9  # 1/Bohr；对 ~6 Bohr 的盒子是安全区间


def _translation_list(nmax: int = 8):
    return [t for t in itertools.product(range(-nmax, nmax + 1), repeat=3)]


def ewald_energy(cell: Cell, charges, eta: float = DEFAULT_ETA, nmax: int = 8):
    lat = cell.lattice
    omega = volume(lat)
    r = positions(cell)  # (nat, 3)
    z = charges
    nat = int(z.shape[0])
    zz = z[:, None] * z[None, :]  # (nat, nat)

    # --- real space
    e_real = jnp.asarray(0.0)
    trans = jnp.asarray(_translation_list(nmax), dtype=lat.dtype)  # (nT, 3)
    tvec = trans @ lat  # (nT, 3)
    for i in range(nat):
        for j in range(nat):
            d = r[i] - r[j] + tvec  # (nT, 3)
            dist = jnp.sqrt(jnp.sum(d * d, axis=-1) + 1e-30)
            val = jax_erfc(eta * dist) / dist
            if i == j:
                # 排除 T = 0（自能由 E_self 承担）
                zero = jnp.all(trans == 0, axis=-1)
                val = jnp.where(zero, 0.0, val)
            e_real = e_real + 0.5 * zz[i, j] * jnp.sum(val)

    # --- reciprocal space
    bmat = 2.0 * jnp.pi * jnp.linalg.inv(lat).T
    m = jnp.asarray(list(itertools.product(range(-nmax - 2, nmax + 3), repeat=3)))
    g = m @ bmat
    g2 = jnp.sum(g * g, axis=-1)
    keep = g2 > 1e-12
    g = g[keep]
    g2 = g2[keep]
    sf = jnp.sum(z[None, :] * jnp.exp(-1j * (g @ r.T)), axis=-1)  # (nG,)
    e_recip = (2.0 * jnp.pi / omega) * jnp.sum(
        jnp.exp(-g2 / (4.0 * eta ** 2)) / g2 * jnp.real(sf * jnp.conj(sf))
    )

    # --- self + background
    e_self = -(eta / jnp.sqrt(jnp.pi)) * jnp.sum(z * z)
    e_bg = -(jnp.pi / (2.0 * omega * eta ** 2)) * jnp.sum(z) ** 2
    return e_real + e_recip + e_self + e_bg


def jax_erfc(x):
    from jax.scipy.special import erfc
    return erfc(x)
