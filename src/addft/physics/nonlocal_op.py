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
from ..geometry.psp import Pseudo, beta_q


def dion_full(psp: Pseudo, spec_idx) -> jnp.ndarray:
    """(nat*nproj, nat*nproj) 分块对角的 D 矩阵。"""
    nat = int(spec_idx.shape[0])
    npj = psp.nproj
    # (nat, nat, npj, npj) 的「块对角」张量：只有 i==j 的块非零
    d = jnp.zeros((nat, nat, npj, npj), dtype=jnp.float64)
    for i in range(nat):
        d = d.at[i, i].set(psp.dion[jnp.asarray(spec_idx)[i]])
    return d.reshape(nat * npj, nat * npj)


def projectors(psp: Pseudo, spec_idx, basis: Basis, cell: Cell, kpts):
    """(nk, nat*nproj, nG) complex —— β_Ii(G)，含结构因子与 1/√Ω。"""
    nat = int(spec_idx.shape[0])
    npj = psp.nproj
    r = positions(cell)  # (nat, 3)
    omega = volume(cell.lattice)
    out = []
    for ik in range(int(kpts.shape[0])):
        gk = gvectors(basis, cell.lattice, kpts[ik])  # (nG, 3)
        q = jnp.sqrt(jnp.sum(gk * gk, axis=-1))  # (nG,)
        # β̃_i(q) 逐原子求值：(nat, nproj, nG)
        bet = jnp.stack([
            beta_q(psp, spec_idx[i], q) for i in range(nat)
        ], axis=0) if nat > 0 else jnp.zeros((0, npj, basis.nG))
        phase = jnp.exp(-1j * (gk @ r.T))  # (nG, nat)
        blk = bet * phase.T[:, None, :]  # (nat, nproj, nG)
        out.append(blk.reshape(nat * npj, basis.nG) / jnp.sqrt(omega))
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
