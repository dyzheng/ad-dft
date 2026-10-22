"""L4 微分层 · SCF 物理映射的线性化（响应算子 A = I − ∂_n F）。

为什么不直接 `jax.jacfwd(scf_map)`：
    在对称构型上 H 有严格简并，eigh 的 JVP 给出 1/(ε_n − ε_m) = 1/0 → NaN。
    （本仓库实测：二聚体对称构型上 AD 力全为 NaN，而有限差分给出 0。
     这是一个「看起来对」的陷阱：只看 FD 会以为力是零所以没问题。）
    这正是方案里 D4 级（简并子空间）的典型失败模式。

这里改为**解析一阶微扰**，并把简并处处理成物理上正确的极限：
    (f_n − f_m)/(ε_n − ε_m) → f'(ε̄)，当 |ε_n − ε_m| < tol
这个替换不是数值技巧，而是占据加权密度矩阵在简并点的连续延拓：
简并子空间内的任何酉旋转给出相同的 δρ（子空间不变性）。

费米能 μ 的标量隐式微分在这里显式出现：
    δμ = Σ w f'(ε) δε / Σ w f'(ε)
漏掉它，电子数不守恒。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..geometry.basis import density_g, g2, psi_to_real, real_from_g
from ..geometry.cell import Cell
from ..physics import xc
from ..system import SCFState, Static

FOUR_PI = 4.0 * jnp.pi


def _full_eigensystem(geo: Cell, static: Static, state: SCFState):
    from ..solver.scf import effective_potential, hamiltonian

    v_r, _ = effective_potential(geo, static, state.rho_r)
    H = hamiltonian(geo, static, v_r)
    eig, vec = jnp.linalg.eigh(H)  # (nk, nG), (nk, nG, nG)（列为本征矢）
    return eig, vec


def make_response(geo: Cell, static: Static, state: SCFState, tol: float = 1e-6):
    """返回纯函数 `apply(dn) -> (∂F/∂n) dn`，简并安全。"""
    from ..solver.scf import vmat_from_real

    basis = static.basis
    nfft = basis.nfft
    lat = geo.lattice
    q2 = g2(basis, lat)
    kxc = xc.xc_kernel_r(state.rho_r)  # (nfft,)
    eig, vec = _full_eigensystem(geo, static, state)
    nband = static.nband
    sigma = static.sigma
    deg = static.spin_degeneracy
    w = static.kweights
    mu = state.mu

    def apply(dn_r):
        # 1) δV_eff = K_H δn + K_xc δn
        ng = density_g(basis, lat, dn_r)
        safe = jnp.where(q2 > 1e-14, q2, 1.0)
        dvg = jnp.where(q2 > 1e-14, FOUR_PI * ng / safe, 0.0)
        dv_r = real_from_g(basis, lat, dvg) + kxc * dn_r

        # 2) δH 矩阵元
        dH = vmat_from_real(basis, dv_r)

        out = jnp.zeros((nfft,), dtype=dn_r.dtype)
        for ik in range(int(state.psi.shape[0])):
            # 关键：本征矢的微扰必须包含向**所有**更高态的跃迁。
            # scf_map 内部对完整 H 做对角化再截取最低 nband，因此低能本征矢会
            # 随势变化向高态旋转；只保留 nband 维子空间会把响应算成 0
            # （实测：完全占据的 2 带子空间内，子空间内旋转不改变密度）。
            C = vec[ik]
            eps = eig[ik]
            M = jnp.conj(C).T @ dH @ C
            M = 0.5 * (M + jnp.conj(M).T)
            deps = jnp.real(jnp.diag(M))

            d = eps[None, :] - eps[:, None]  # d[m,n] = ε_n − ε_m
            off = jnp.abs(d) > tol
            wgt = jnp.where(off, 1.0 / jnp.where(off, d, 1.0), 0.0)
            coef = M * wgt  # coef[m,n] = M_mn/(ε_n−ε_m)

            f = jnp.concatenate([state.occ[ik], jnp.zeros(eig.shape[-1] - nband)])
            if sigma > 1e-6:
                fprime = f * (1.0 - f) / sigma
                denom = jnp.sum(w[ik] * fprime) + 1e-30
                dmu = jnp.sum(w[ik] * fprime * deps) / denom
                df = fprime * (deps - dmu)
            else:  # 整数占据：占据数不随 ε 变，响应项为零
                df = jnp.zeros_like(f)

            dc = (C @ coef)[:, :nband].T  # (nband, nG)
            psi_r = psi_to_real(basis, lat, state.psi[ik])
            dpsi_r = psi_to_real(basis, lat, dc)
            dmod2 = 2.0 * jnp.real(jnp.conj(psi_r) * dpsi_r)
            out = out + w[ik] * deg * (
                jnp.sum(f[:nband, None] * dmod2, axis=0)
                + jnp.sum(df[:nband, None] * jnp.abs(psi_r) ** 2, axis=0)
            )
        return out

    return apply


def build_jacobian(geo: Cell, static: Static, state: SCFState) -> jnp.ndarray:
    """把响应算子展开成稠密矩阵（V0 小体系）。生产路径用 matrix-free（K12）。"""
    apply = make_response(geo, static, state)
    eye = jnp.eye(static.basis.nfft)
    return jax.vmap(apply)(eye).T  # 第 j 列 = J e_j
