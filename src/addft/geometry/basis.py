"""L1 基组 · 冻结壳层 + 与 ABACUS 同规则的 FFT 盒 (spec/discretization.md).

两件事必须分清：

1. **整数指标集合 `miller`**——在参考构型一次性确定，之后冻结。这是「壳层冻结」策略
   （方案 §2.2 契约一，选 (a)）。冻结的不是能量，是「哪些 G 存在」。
2. **物理 G 矢量**——`gvectors(basis, lattice)` 每次都从 lattice 重算，
   于是 G → (1+ε)^{-T}G、Ω → det(I+ε)Ω 两条链自动进入 AD 图，Pulay 项不会漏。

FFT 盒规则与 ABACUS `source_basis/module_pw/pw_init.cpp:92-137` 对齐：
先取 `ibox = 2*max|m_i| + 1`（奇数），再自增到只含 2、3、5 因子的最小整数。
ABACUS 源码里 7 的分支被注释掉，因此这里同样不允许 7。
密度网格默认 `ecutrho = 4 * ecutwfc`（ABACUS read_inp_sys.cpp:1227）。
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ..geometry.cell import reciprocal, volume


class Basis(NamedTuple):
    miller: jnp.ndarray  # (nG, 3) int32 —— 冻结的整数指标
    ecut: float  # Ha, 动能判据 |k+G|^2 / 2 <= ecut
    fft_shape: tuple  # 密度/波函数共用的实空间盒

    @property
    def nG(self) -> int:
        return int(self.miller.shape[0])

    @property
    def nfft(self) -> int:
        return int(np.prod(self.fft_shape))


def shell_radius2(ecut: float) -> float:
    """动能截断对应的 |G|^2 上界（原子单位：T = |G|^2/2）。"""
    return 2.0 * float(ecut)


def is_smooth235(n: int) -> bool:
    if n < 1:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


def smooth_up(n: int) -> int:
    """向上取最近的 2·3·5 光滑数（ABACUS 的 ibox 自增逻辑）。"""
    while not is_smooth235(n):
        n += 1
    return n


def _nmax_on_axis(lattice, gcut2: float) -> tuple[int, int, int]:
    """每个倒格方向上需要多少个整数指标才能覆盖球 |G|^2 <= gcut2。"""
    lat = np.asarray(lattice, dtype=float)
    a_len = np.linalg.norm(lat, axis=1)  # |a_i|
    # |m_i b_i| >= |m_i| * 2π/|a_i|，故 m_i 的保守上界：
    m = np.ceil(np.sqrt(gcut2) * a_len / (2.0 * np.pi)).astype(int)
    return int(m[0]), int(m[1]), int(m[2])


def build_miller(lattice, ecut: float, gamma_only: bool = True) -> jnp.ndarray:
    """在**参考构型**上确定整数指标集合。此函数不参与微分。"""
    gcut2 = shell_radius2(ecut)
    lat = np.asarray(lattice, dtype=float)
    bmat = 2.0 * np.pi * np.linalg.inv(lat).T
    m1, m2, m3 = _nmax_on_axis(lattice, gcut2)
    rng = np.stack(np.meshgrid(
        np.arange(-m1, m1 + 1),
        np.arange(-m2, m2 + 1),
        np.arange(-m3, m3 + 1),
        indexing="ij",
    ), axis=-1).reshape(-1, 3)
    cart = rng @ bmat
    keep = np.sum(cart ** 2, axis=1) <= gcut2 * (1.0 + 1e-12)
    if gamma_only:
        pass  # Γ 点：实波函数可用 r2c；此处保留全复表示，由 kernel 层优化
    miller = rng[keep]
    # 排序：先 G=0，再按 |G|^2 升序（决定性顺序，便于跨实现逐元素对比）
    g2 = np.sum((miller @ bmat) ** 2, axis=1)
    order = np.lexsort((miller[:, 2], miller[:, 1], miller[:, 0], g2))
    return jnp.asarray(miller[order], dtype=jnp.int32)


def fft_shape(lattice, ecut: float, density_factor: float = 4.0) -> tuple:
    """按 ABACUS 规则定 FFT 盒：奇数 → 2·3·5 光滑。"""
    gcut2 = shell_radius2(ecut) * float(density_factor)
    m1, m2, m3 = _nmax_on_axis(lattice, gcut2)
    dims = []
    for m in (m1, m2, m3):
        ibox = 2 * m + 1
        dims.append(smooth_up(ibox))
    return tuple(dims)


def make_basis(lattice, ecut: float, density_factor: float = 4.0) -> Basis:
    return Basis(
        miller=build_miller(lattice, ecut),
        ecut=float(ecut),
        fft_shape=fft_shape(lattice, ecut, density_factor),
    )


def gvectors(basis: Basis, lattice, kpoint=None):
    """(nG, 3) 笛卡尔 G(+k)。对 lattice 可微——这是 Pulay 项的来源。"""
    g = basis.miller.astype(lattice.dtype) @ reciprocal(lattice)
    if kpoint is not None:
        g = g + kpoint[None, :]
    return g


def g2(basis: Basis, lattice, kpoint=None):
    g = gvectors(basis, lattice, kpoint)
    return jnp.sum(g * g, axis=-1)


def shell_margin(basis: Basis, lattice) -> float:
    """冻结壳层的安全裕度：最外层 |G|^2 与截断面的相对距离。

    验收要求（方案 §2.2）：壳层跳变处的应力不连续 < 0.1 kbar，
    因此 `shell_margin` 必须 > 5%，否则该算例不允许进入应力校验集。
    """
    r2 = g2(basis, lattice)
    return float((shell_radius2(basis.ecut) - r2.max()) / shell_radius2(basis.ecut))


# --------------------------------------------------------------------------
# G 空间 <-> 实空间（1/√Ω 约定，见 spec/units.md）
# --------------------------------------------------------------------------

def _flat_index(basis: Basis) -> jnp.ndarray:
    n1, n2, n3 = basis.fft_shape
    m = basis.miller
    i1 = jnp.mod(m[:, 0], n1)
    i2 = jnp.mod(m[:, 1], n2)
    i3 = jnp.mod(m[:, 2], n3)
    return (i1 * n2 + i2) * n3 + i3


def coeffs_to_grid(basis: Basis, c: jnp.ndarray) -> jnp.ndarray:
    """(..., nG) -> (..., n1, n2, n3) 的零填充谱数组。"""
    lead = c.shape[:-1]
    flat = jnp.zeros(lead + (basis.nfft,), dtype=c.dtype)
    flat = flat.at[..., _flat_index(basis)].set(c)
    return flat.reshape(lead + tuple(basis.fft_shape))


def grid_to_coeffs(basis: Basis, arr: jnp.ndarray) -> jnp.ndarray:
    """(..., n1, n2, n3) -> (..., nG)。"""
    lead = arr.shape[:-3]
    flat = arr.reshape(lead + (basis.nfft,))
    return jnp.take(flat, _flat_index(basis), axis=-1)


def psi_to_real(basis: Basis, lattice, c: jnp.ndarray) -> jnp.ndarray:
    """ψ(r) = (1/√Ω) Σ_G c_G e^{iG·r}；jnp.fft 归一化补偿 N/√Ω。输出 (..., nfft) 扁平。"""
    n = basis.nfft
    omega = volume(lattice)
    grid = coeffs_to_grid(basis, c)
    out = jnp.fft.ifftn(grid, axes=(-3, -2, -1)) * (n / jnp.sqrt(omega))
    return out.reshape(c.shape[:-1] + (n,))


def real_to_psi(basis: Basis, lattice, psi_r: jnp.ndarray) -> jnp.ndarray:
    """c_G = (1/√Ω) ∫ ψ e^{-iG·r}；离散补偿 √Ω/N。输入 (..., nfft) 扁平。"""
    n = basis.nfft
    omega = volume(lattice)
    grid = psi_r.reshape(psi_r.shape[:-1] + tuple(basis.fft_shape))
    spec = jnp.fft.fftn(grid, axes=(-3, -2, -1)) * (jnp.sqrt(omega) / n)
    return grid_to_coeffs(basis, spec)


def density_g(basis: Basis, lattice, rho_r: jnp.ndarray) -> jnp.ndarray:
    """n(G) = (1/Ω)∫n e^{-iG·r} = fftn(n)/N。输入 (nfft,) 扁平。"""
    n = basis.nfft
    grid = rho_r.reshape(tuple(basis.fft_shape))
    spec = jnp.fft.fftn(grid) / n
    return grid_to_coeffs(basis, spec)


def real_from_g(basis: Basis, lattice, vg: jnp.ndarray) -> jnp.ndarray:
    """v(r) = Σ_G v_G e^{iG·r} = N · ifftn(v_grid)；输出扁平 (nfft,)。"""
    grid = coeffs_to_grid(basis, vg)
    return (basis.nfft * jnp.fft.ifftn(grid)).real.reshape(basis.nfft)
