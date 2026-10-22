"""L3 求解器 · 哈密顿量组装、SCF 不动点与混合器。

分层的关键约定：
  * 混合器 (mixer) 是**加速器不是物理**，永远不进微分图；
  * SCF 的「物理映射」`scf_map` 才是反向传播的对象：n ↦ 对角化 ↦ 占据 ↦ 密度；
  * 收敛判据用**归一化密度残差**（与 ABACUS chg_drho.cpp 同口径），不用能量差。
    能量差小但密度残差大时，梯度不可信。

目标函数（显式泛函，不含 double counting）：
    E = T[ψ] + ∫v_loc n + E_H[n] + E_xc[n] + E_NL[ψ] + E_II
这个形式的好处是：每个 term 都是 n 与 ψ 的显式函数，力的三项 Pulay 全由链式法则给出。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..geometry.basis import Basis, g2, gvectors
from ..geometry.cell import Cell, volume
from ..physics import hartree, xc
from ..physics.density import density_from_psi
from ..physics.local import local_potential_r
from ..physics.occupation import occupations
from ..system import SCFState, Static
from .eigh import eigh_lowest


# ------------------------------------------------------------------ 哈密顿量

def vmat_from_real(basis: Basis, v_r):
    """<G'|V|G> = fft(V)[(m_G − m_G') mod N] / N。实空间势 → 稠密矩阵，无额外约定。"""
    n1, n2, n3 = basis.fft_shape
    vg = jnp.fft.fftn(v_r.reshape(n1, n2, n3)) / basis.nfft
    m = basis.miller
    d = m[:, None, :] - m[None, :, :]
    i1 = jnp.mod(d[..., 0], n1)
    i2 = jnp.mod(d[..., 1], n2)
    i3 = jnp.mod(d[..., 2], n3)
    idx = (i1 * n2 + i2) * n3 + i3
    return vg.reshape(-1)[idx]


def kinetic_diag(basis: Basis, lattice, kpt):
    kg = gvectors(basis, lattice, kpt)
    return 0.5 * jnp.sum(kg * kg, axis=-1)


def hamiltonian(geo: Cell, static: Static, v_r):
    """返回 (nk, nG, nG) 的 H = T + V_loc + V_H + V_xc + V_NL。

    V_NL 必须出现在 H 里——否则本征值缺非局域项，而能量分解里又有它，
    两者不自洽（实测差 0.046 Ha，正好等于 E_NL）。这是「能量对、梯度错」的
    另一种形态：两种能量写法不一致。
    """
    from ..physics import nonlocal_op

    basis = static.basis
    vm = vmat_from_real(basis, v_r)
    proj = nonlocal_op.projectors(static.psp, static.spec_idx, basis, geo, static.kpts)
    d = nonlocal_op.dion_full(static.psp, static.spec_idx)
    hs = []
    for ik in range(int(static.kpts.shape[0])):
        t = kinetic_diag(basis, geo.lattice, static.kpts[ik])
        vnl = jnp.einsum("pg,pq,qh->gh", proj[ik], d, jnp.conj(proj[ik]))
        h = jnp.diag(t) + vm + vnl
        hs.append(0.5 * (h + jnp.conj(h.T)))  # 显式 Hermitian 化（数值）
    return jnp.stack(hs, axis=0)


def effective_potential(geo: Cell, static: Static, rho_r, with_xc=True):
    basis = static.basis
    v_loc = local_potential_r(static.psp, static.spec_idx, basis, geo)
    v_h = hartree.hartree_potential_r(basis, geo.lattice, rho_r)
    if with_xc:
        v_xc = xc.xc_potential_r(xc.xc_energy_lda, basis, geo.lattice, rho_r)
    else:
        v_xc = jnp.zeros_like(v_h)
    return v_loc + v_h + v_xc, (v_loc, v_h, v_xc)


# ------------------------------------------------------------------ 物理映射

def scf_map(geo: Cell, static: Static, rho_r):
    """一次物理映射：n → H[n] → 对角化 → 占据 → n'。

    这是反向传播唯一允许看到的映射：混合器不在这里。
    """
    v_r, _ = effective_potential(geo, static, rho_r)
    H = hamiltonian(geo, static, v_r)
    eig, vec = eigh_lowest(H, static.nband)
    occ, mu = occupations(eig, static.kweights, static.nelec, static.sigma,
                          static.spin_degeneracy)
    return density_from_psi(static.basis, geo.lattice, vec, occ,
                            static.kweights, static.spin_degeneracy)


# ------------------------------------------------------------------ 混合器

def q2_grid(basis: Basis, lattice):
    """全 FFT 网格上的 |G|^2（用于预条件子与滤波，不进微分图）。"""
    n1, n2, n3 = basis.fft_shape
    i1 = jnp.arange(n1); i2 = jnp.arange(n2); i3 = jnp.arange(n3)
    m1 = jnp.where(i1 < n1 // 2, i1, i1 - n1)
    m2 = jnp.where(i2 < n2 // 2, i2, i2 - n2)
    m3 = jnp.where(i3 < n3 // 2, i3, i3 - n3)
    mm = jnp.stack(jnp.meshgrid(m1, m2, m3, indexing="ij"), axis=-1).astype(jnp.float64)
    bmat = 2.0 * jnp.pi * jnp.linalg.inv(lattice).T
    g = mm @ bmat
    return jnp.sum(g * g, axis=-1)


def kerker_mix(basis: Basis, lattice, rho_in, rho_out, alpha=0.5, gg0=1.0):
    """Kerker 预条件 + 线性混合（G 空间）。只影响正向收敛速度，不进微分图。"""
    shape = tuple(basis.fft_shape)
    dn = (rho_out - rho_in).reshape(shape)
    dg = jnp.fft.fftn(dn) / basis.nfft
    q2 = q2_grid(basis, lattice)
    filt = jnp.where(q2 > 1e-12, q2 / (q2 + gg0), 0.0)
    dn_f = (basis.nfft * jnp.fft.ifftn(dg * filt)).real.reshape(basis.nfft)
    return rho_in + alpha * dn_f


def density_residual(basis: Basis, lattice, rho_new, rho_old, nelec):
    """ABACUS scf_thr_type=2 同口径：Σ|Δρ|·Ω/N / N_e。"""
    omega = volume(lattice)
    return jnp.sum(jnp.abs(rho_new - rho_old)) * omega / basis.nfft / nelec


# ------------------------------------------------------------------ SCF 驱动

def initial_density(static: Static, geo: Cell):
    omega = volume(geo.lattice)
    return jnp.full((static.basis.nfft,), static.nelec / omega, dtype=jnp.float64)


def solve_scf(geo: Cell, static: Static, alpha=0.5, gg0=1.0,
              tol=1e-10, max_iter=200, verbose=False, mixer="diis", beta=0.5):
    """正向求解。混合器在这里，但**不在** scf_map 里。"""
    from .mixer import DIISMixer

    basis = static.basis
    rho = initial_density(static, geo)
    mix = DIISMixer(basis.nfft, m=8, beta=beta) if mixer == "diis" else None
    res = jnp.asarray(1.0)
    prev = 1e9
    it = 0
    for it in range(1, max_iter + 1):
        rho_out = scf_map(geo, static, rho)
        res = density_residual(basis, geo.lattice, rho_out, rho, static.nelec)
        if mixer == "diis":
            if float(res) > prev * 1.5:      # 振荡检测 → restart 历史
                mix.reset()
            prev = float(res)
            rho = mix(rho, rho_out)
        else:
            rho = kerker_mix(basis, geo.lattice, rho, rho_out, alpha, gg0)
        if verbose and (it % 10 == 0 or it < 3):
            print(f"  scf {it:3d}  drho={float(res):.3e}")
        if float(res) < tol:
            break
    # 收敛点"抛光"：把混合后的 ρ 直接代回物理映射，直到它是严格不动点。
    # 为什么必须做：混合器给出的 ρ 与 F(ρ) 还差一个残差量级的差，此时
    # ψ 是 H[ρ] 的本征矢，但 ρ ≠ Σf|ψ|²。包络定理（dE/dθ = ∂_θE|_ψ）要求两者
    # 严格一致；否则"响应项应该为零"会退化成 1e-3 量级的假信号。
    # 实测：抛光前 d_state 与有限差分差 2.3e-6（相对），抛光后见 tests/。
    for _ in range(4):
        rho_new = scf_map(geo, static, rho)
        res = density_residual(basis, geo.lattice, rho_new, rho, static.nelec)
        rho = rho_new
        if float(res) < 1e-14:
            break

    v_r, _ = effective_potential(geo, static, rho)
    H = hamiltonian(geo, static, v_r)
    eig, vec = eigh_lowest(H, static.nband)
    occ, mu = occupations(eig, static.kweights, static.nelec, static.sigma,
                          static.spin_degeneracy)
    return SCFState(psi=vec, eig=eig, occ=occ, mu=mu, rho_r=rho,
                    n_iter=jnp.asarray(it), residual=jnp.asarray(res))
