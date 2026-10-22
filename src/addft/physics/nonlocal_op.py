"""L2 原语 · 非局域赝势投影（方案 §4.4 的「投影必须从几何生成」）。

V_NL = Σ_I Σ_ij |β_Ii> D_Iij <β_Ij|
β_Ii(G) = (1/√Ω) β̃_i(|k+G|) e^{-i(k+G)·R_I}

三件事在这里同时发生，缺一即错：
  * β̃_i(|k+G|)：径向插值必须是 (R, cell) 的可微函数 → 每次几何变化重建；
  * e^{-i(k+G)·R_I}：符号约定必须与 local.py 一致（同为 e^{-iG·R}）；
  * 1/√Ω：指数必须是 −1/2（应力体积项系数由此而来）。

测试三件套：Hermitian 性、伴随恒等式 ⟨Ax,y⟩=⟨x,A^T y⟩（到机器精度）、
对 R 的 VJP 与中心差分一致（三个构型）。
"""

from __future__ import annotations

import jax.numpy as jnp

from ..geometry.basis import Basis, g2, gvectors
from ..geometry.cell import Cell, positions, volume
from ..geometry.psp import Pseudo, beta_q, lm_table
from .ylm import ylm_real


def dion_full(psp: Pseudo, spec_idx) -> jnp.ndarray:
    """(nat*nh, nat*nh) 的块对角 D，nh=Σ(2l+1)。

    关键：块索引必须放在 (i, p, j, q)，reshape 后全局行=(i,p)、列=(j,q)。
    不能写成 (i, j, p, q) —— 那会把不同原子的块交错成非 Hermitian 矩阵。
    径向 D 对同一 l 的全部 m 通道复制；不同 (l,m) 之间为零。
    """
    si = jnp.asarray(spec_idx)
    nat = int(si.shape[0])
    nh = int(psp.nh[int(si[0])])
    d = jnp.zeros((nat, nh, nat, nh), dtype=psp.dion.dtype)
    lm = lm_table(psp.proj_l[int(si[0])])
    for i in range(nat):
        for p, (ib, l, m) in enumerate(lm):
            for q, (jb, l2, m2) in enumerate(lm):
                if l == l2 and m == m2:
                    d = d.at[i, p, i, q].set(psp.dion[int(si[i]), ib, jb])
    return d.reshape(nat * nh, nat * nh)


def projectors(psp: Pseudo, spec_idx, basis: Basis, cell: Cell, kpts):
    """(nk, nat*nh, nG) complex —— 每个径向 β 展开为 m=0..2l 个通道。"""
    nat = int(spec_idx.shape[0])
    nh = int(psp.nh[int(spec_idx[0])])  # 本轮单 species
    r = positions(cell)  # (nat, 3)
    omega = volume(cell.lattice)
    out = []
    for ik in range(int(kpts.shape[0])):
        gk = gvectors(basis, cell.lattice, kpts[ik])  # (nG, 3)
        q = jnp.sqrt(jnp.sum(gk * gk, axis=-1))  # (nG,)
        ylm = ylm_real(int(psp.lmax[int(spec_idx[0])]), gk)  # (nG,(lmax+1)^2)
        phase = jnp.exp(-1j * (gk @ r.T))  # (nG, nat)
        rows = []
        for i in range(nat):
            si = int(spec_idx[i])
            bet = beta_q(psp, si, q)  # (nrad,nG)
            for ib, l, m in lm_table(psp.proj_l[si]):
                rows.append(bet[ib] * ylm[:, l * l + m] * phase[:, i])
        out.append(jnp.stack(rows, axis=0) / jnp.sqrt(omega))
    return jnp.stack(out, axis=0)


def becp(proj, psi):
    """<β|ψ>：proj (nk, nproj_tot, nG), psi (nk, nband, nG) → (nk, nproj_tot, nband)。"""
    return jnp.einsum("kpg,kng->kpn", jnp.conj(proj), psi)


def vnl_apply(proj, dion, psi):
    """V_NL ψ 的矩阵-free 实现（两次 GEMM，与 ABACUS nonlocal_workspace 同构）。"""
    b = becp(proj, psi)
    scaled = jnp.einsum("pq,kqn->kpn", dion, b)
    return jnp.einsum("kpg,kpn->kng", proj, scaled)


def nonlocal_energy(proj, dion, psi, occ, weights, spin_degeneracy=2.0):
    b = becp(proj, psi)
    e = jnp.einsum("kpn,pq,kqn->kn", jnp.conj(b), dion, b).real
    return jnp.sum(weights[:, None] * occ * e) * spin_degeneracy
