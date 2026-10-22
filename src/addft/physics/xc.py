"""L2 原语 · 交换关联：**只给能量密度，势由 AD 自动产生** (方案 §4.3)。

这是全项目最重要的一个原语设计决定，理由是它带来两个可验证的收益：

1. 对 GGA，v_xc = ∂f/∂n − ∇·(∂f/∂∇n) 里的 −∇· 项**不需要手写**。
   只要把 E_xc 写成「网格值 n_i 的标量函数」，并且 ∇n 由 n_i 经线性算子得到，
   链式法则自动给出 −∇·；而手写这一项正是传统代码里 GGA 最容易出错的地方
   （符号、边界、离散化不匹配）。
2. 它把「能量对」与「梯度对」的耦合变成构造性的：不可能出现「能量公式对、
   势是另写一份」的分叉。

代价是逐点建图太慢，因此生产路径把 (E, v) 融进单个 custom_vjp kernel（K07）。
本文件的 `xc_potential_r` 就是那条自定义 kernel 的**语义定义**（ref 级）。
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

try:
    from .libxc_bridge import libxc_lda_energy_density, libxc_lda_kernel
    _HAVE_LIBXC = True
except Exception:  # pragma: no cover - 取决于系统是否装了 libxc
    _HAVE_LIBXC = False

USE_LIBXC = os.environ.get("ADDFT_XC", "libxc") != "analytic"
if USE_LIBXC and not _HAVE_LIBXC:
    raise ImportError("找不到 libxc；设置 ADDFT_XC=analytic 使用解析 PZ81 fallback")

XC_LDA_X = 1
XC_LDA_C_PZ = 9

from ..geometry.basis import Basis, density_g, gvectors, real_from_g
from ..geometry.cell import volume

# ---------------------------------------------------------------- LDA (PZ81)

_A, _B, _C, _D = 0.0311, -0.048, 0.0020, -0.0116  # Ha, rs < 1
_G, _B1, _B2 = -0.1423, 1.0529, 0.3334  # Ha, rs >= 1


def _eps_x(n):
    """Dirac 交换能密度（Ha/粒子）。"""
    n = jnp.maximum(n, 1e-14)
    return -(3.0 / 4.0) * jnp.power(3.0 * n / jnp.pi, 1.0 / 3.0)


def _eps_c(n):
    """Perdew–Zunger 81 拟合的 LDA 关联能（Ha/粒子）。"""
    n = jnp.maximum(n, 1e-14)
    rs = jnp.power(3.0 / (4.0 * jnp.pi * n), 1.0 / 3.0)
    small = _A * jnp.log(rs) + _B + _C * rs * jnp.log(rs) + _D * rs
    large = _G / (1.0 + _B1 * jnp.sqrt(rs) + _B2 * rs)
    return jnp.where(rs < 1.0, small, large)


def lda_energy_density(n):
    """n · ε_xc(n)，单位 Ha/Bohr³。"""
    n = jnp.maximum(n, 1e-14)
    return n * (_eps_x(n) + _eps_c(n))


# ------------------------------------------------------- 谱式 ∇ 与 ∇·（GGA 用）

def grad_r(basis: Basis, lattice, f_r):
    """∇f 的谱式实现，返回 (3, nfft)。"""
    fg = density_g(basis, lattice, f_r)
    ig = 1j * gvectors(basis, lattice)  # (nG, 3)
    out = []
    for alpha in range(3):
        out.append(real_from_g(basis, lattice, ig[:, alpha] * fg))
    return jnp.stack(out, axis=0)


def div_r(basis: Basis, lattice, v_r):
    """∇·v 的谱式实现；满足 ⟨∇f, v⟩ = −⟨f, ∇·v⟩（离散伴随恒等式）。"""
    vg = jnp.stack([density_g(basis, lattice, v_r[a]) for a in range(3)], axis=-1)
    ig = 1j * gvectors(basis, lattice)
    total = jnp.sum(ig * vg, axis=-1)
    return real_from_g(basis, lattice, total)


# ------------------------------------------------------------------ 能量泛函

def lda_x_energy_density(n):
    n = jnp.maximum(n, 1e-14)
    return n * _eps_x(n)


def lda_c_energy_density(n):
    n = jnp.maximum(n, 1e-14)
    return n * _eps_c(n)


def _analytic_lda_energy(basis, lattice, rho_r):
    return (volume(lattice) / basis.nfft) * jnp.sum(lda_energy_density(rho_r))


def xc_energy_lda(basis: Basis, lattice, rho_r):
    if not USE_LIBXC:
        return _analytic_lda_energy(basis, lattice, rho_r)
    e = libxc_lda_energy_density(rho_r, XC_LDA_X)
    e = e + libxc_lda_energy_density(rho_r, XC_LDA_C_PZ)
    return (volume(lattice) / basis.nfft) * jnp.sum(e)


def exchange_energy_lda(basis: Basis, lattice, rho_r):
    if not USE_LIBXC:
        return (volume(lattice) / basis.nfft) * jnp.sum(lda_x_energy_density(rho_r))
    e = libxc_lda_energy_density(rho_r, XC_LDA_X)
    return (volume(lattice) / basis.nfft) * jnp.sum(e)


def correlation_energy_lda(basis: Basis, lattice, rho_r):
    if not USE_LIBXC:
        return (volume(lattice) / basis.nfft) * jnp.sum(lda_c_energy_density(rho_r))
    e = libxc_lda_energy_density(rho_r, XC_LDA_C_PZ)
    return (volume(lattice) / basis.nfft) * jnp.sum(e)


def gga_model_energy_density(n, sigma):
    """GGA 模型泛函（非 PBE，仅用于验证 ∇/−∇· 的转置关系；K07 换 libxc）。

    ε = ε_LDA(n) · (1 + μ s²)，s = |∇n| / (2 k_F n)。
    """
    n = jnp.maximum(n, 1e-14)
    kf = jnp.power(3.0 * jnp.pi ** 2 * n, 1.0 / 3.0)
    s2 = jnp.maximum(sigma, 0.0) / (4.0 * kf ** 2 * n ** 2)
    return lda_energy_density(n) * (1.0 + 0.1234 * s2)


def xc_energy_gga(basis: Basis, lattice, rho_r):
    g = grad_r(basis, lattice, rho_r)  # (3, nfft)
    sigma = jnp.sum(g * g, axis=0)
    omega = volume(lattice)
    return (omega / basis.nfft) * jnp.sum(gga_model_energy_density(rho_r, sigma))


def xc_potential_r(xc_energy_fn, basis: Basis, lattice, rho_r):
    """v_xc = δE_xc/δn —— 由 AD 生成，不手写 −∇· 项。

    **离散泛函导数的权重是契约的一部分**：E_xc 离散化为 (Ω/N)Σ_j f(n_j)，
    因此 jax.grad(E_xc) 给出的是 (Ω/N)·f'(n_j)，而物理势是 f'(n_j)。
    两者相差 N/Ω —— 漏掉它，势小 8 倍（本仓库实测：η=1728/N 时 XC 势弱 8×，
    能量仍然"看起来合理"，但驻点性被破坏，力偏差 25%）。
    这一类错误只有「能量对密度的驻点性」测试能抓住：∂E/∂n|_{ψ} 必须为 0。

    对**整个密度场**求导，因此 GGA 的 −∇· 项会自动出现（方案 §4.3 的断言）。
    """
    n = basis.nfft
    omega = volume(lattice)
    g = jax.grad(lambda r: xc_energy_fn(basis, lattice, r))(rho_r)
    return g * (n / omega)


def xc_potential_point(n):
    """单点 v_xc(n) = d[n·ε(n)]/dn —— 「energy-only」接口的点级定义。"""
    return jax.grad(lambda x: x * (_eps_x(x) + _eps_c(x)))(n)


def xc_kernel_point(n):
    """单点 K_xc(n) = d²[n·ε(n)]/dn² —— 响应算子里的对角核。"""
    return jax.grad(xc_potential_point)(n)


def xc_kernel_r(n):
    """整个网格上的对角核，一次 host callback，避免对 callback 做 vmap。"""
    if not USE_LIBXC:
        return jax.vmap(xc_kernel_point)(n)
    return (libxc_lda_kernel(n, XC_LDA_X) +
            libxc_lda_kernel(n, XC_LDA_C_PZ))
