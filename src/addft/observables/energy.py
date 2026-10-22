"""L5 观测量 · 能量分解（**带 tag，可逐项开关**）。

`tag` 不是为了好看，它是第 6 章「项级消融」的物质基础：
不能单独关掉的项，就不能被测量；不能被测量的项，推导对不对只能靠祈祷。

ablation 的用法：
    energy_terms(geo, static, state, active={"kinetic","hartree"})
harness 自动跑「全开 / 逐个关 / 两两关」，输出贡献矩阵，识别「相消对」。
"""

from __future__ import annotations

import jax.numpy as jnp

from ..geometry.basis import gvectors
from ..geometry.cell import Cell, volume
from ..physics import hartree, nonlocal_op, xc
from ..physics.ewald import ewald_energy
from ..physics.local import local_energy
from ..system import Static, SCFState

ALL_TAGS = ("kinetic", "local", "hartree", "exchange", "correlation", "nonlocal", "ewald")


def _kinetic(geo, static, state):
    total = 0.0
    for ik in range(int(static.kpts.shape[0])):
        kg = gvectors(static.basis, geo.lattice, static.kpts[ik])
        t = 0.5 * jnp.sum(kg * kg, axis=-1)  # (nG,)
        w = static.kweights[ik] * state.occ[ik] * static.spin_degeneracy  # (nband,)
        per_band = jnp.sum(jnp.abs(state.psi[ik]) ** 2 * t[None, :], axis=-1)  # (nband,)
        total = total + jnp.sum(w * per_band)
    return total


def energy_terms(geo: Cell, static: Static, state: SCFState, active=ALL_TAGS):
    """返回 dict[tag] -> scalar (Ha)。只计算 active 里的项。"""
    basis = static.basis
    out = {}
    if "kinetic" in active:
        out["kinetic"] = _kinetic(geo, static, state)
    if "local" in active:
        out["local"] = local_energy(static.psp, static.spec_idx, basis, geo, state.rho_r)[0]
    if "hartree" in active:
        out["hartree"] = hartree.hartree_energy(basis, geo.lattice, state.rho_r)
    if "exchange" in active:
        out["exchange"] = xc.exchange_energy_lda(basis, geo.lattice, state.rho_r)
    if "correlation" in active:
        out["correlation"] = xc.correlation_energy_lda(basis, geo.lattice, state.rho_r)
    if "nonlocal" in active:
        proj = nonlocal_op.projectors(static.psp, static.spec_idx, basis, geo, static.kpts)
        d = nonlocal_op.dion_full(static.psp, static.spec_idx)
        out["nonlocal"] = nonlocal_op.nonlocal_energy(
            proj, d, state.psi, state.occ, static.kweights, static.spin_degeneracy)
    if "ewald" in active:
        charges = static.psp.zval[static.spec_idx]
        out["ewald"] = ewald_energy(geo, charges, eta=static.eta_ewald)
    return out


def total_energy(geo: Cell, static: Static, state: SCFState, active=ALL_TAGS):
    terms = energy_terms(geo, static, state, active)
    e = jnp.asarray(0.0)
    for v in terms.values():
        e = e + v
    return e


def rho_dependent_energy(geo: Cell, static: Static, rho_r, active=("local", "hartree", "exchange", "correlation")):
    """只依赖密度的能量部分——隐式微分里 ∂L/∂n 只需要这一半。"""
    basis = static.basis
    e = jnp.asarray(0.0)
    if "local" in active:
        e = e + local_energy(static.psp, static.spec_idx, basis, geo, rho_r)[0]
    if "hartree" in active:
        e = e + hartree.hartree_energy(basis, geo.lattice, rho_r)
    if "exchange" in active:
        e = e + xc.exchange_energy_lda(basis, geo.lattice, rho_r)
    if "correlation" in active:
        e = e + xc.correlation_energy_lda(basis, geo.lattice, rho_r)
    return e


# --------------------------------------------------------------------------
# 变分形式（生产路径）
# --------------------------------------------------------------------------
def band_energy(geo: Cell, static: Static, state: SCFState):
    """Σ_nk w f <ψ|H|ψ>。不经过 eigh，因此不需要对本征值求导。"""
    from ..solver.scf import effective_potential, hamiltonian

    v_r, _ = effective_potential(geo, static, state.rho_r)
    H = hamiltonian(geo, static, v_r)
    e = jnp.asarray(0.0)
    for ik in range(int(static.kpts.shape[0])):
        hpsi = jnp.einsum("mn,bn->bm", H[ik], state.psi[ik])
        e = e + static.kweights[ik] * jnp.sum(
            state.occ[ik] * static.spin_degeneracy * jnp.real(
                jnp.sum(jnp.conj(state.psi[ik]) * hpsi, axis=-1)))
    return e


def expect_matrix(geo: Cell, static: Static, state: SCFState, v_r):
    """Σ_nk w f <ψ|V|ψ>，用与 H 完全相同的矩阵构造（不做实空间积分）。

    为什么不用 (Ω/N)Σ v·ρ：两者在 v 非带限时并不相等（v_xc 是 n 的逐点非线性
    函数，含到 Nyquist 的频率）。混用会让能量对密度不驻定，力的偏差可达 25%。
    统一走矩阵后，驻点性由构造保证。
    """
    from ..solver.scf import vmat_from_real

    vm = vmat_from_real(static.basis, v_r)
    e = jnp.asarray(0.0)
    for ik in range(int(static.kpts.shape[0])):
        hpsi = jnp.einsum("mn,bn->bm", vm, state.psi[ik])
        e = e + static.kweights[ik] * jnp.sum(
            state.occ[ik] * static.spin_degeneracy
            * jnp.real(jnp.sum(jnp.conj(state.psi[ik]) * hpsi, axis=-1)))
    return e


def energy_variational(geo: Cell, static: Static, state: SCFState):
    """E = Σ f<ψ|H|ψ> − E_H + E_xc − ∫v_xc n + E_II。

    这个形式在收敛点是**驻点**（对 ψ 与 n 都驻定），因此
        dE/dθ = ∂E/∂θ|_(ψ,n)
    对一切 θ 成立（包络定理）。这是「力不需要专门代码」的严格内容；
    显式泛函形式在固定 n 下对 ψ 不驻定，会丢掉一项（实测偏差 ~20%）。
    """
    from ..physics import hartree, xc
    from ..physics.ewald import ewald_energy

    basis = static.basis
    e_band = band_energy(geo, static, state)
    e_h = hartree.hartree_energy(basis, geo.lattice, state.rho_r)
    e_xc = xc.xc_energy_lda(basis, geo.lattice, state.rho_r)
    v_xc = xc.xc_potential_r(xc.xc_energy_lda, basis, geo.lattice, state.rho_r)
    e_dc = expect_matrix(geo, static, state, v_xc)  # = <ψ|V_xc|ψ>，与 H 同源
    charges = static.psp.zval[static.spec_idx]
    e_ii = ewald_energy(geo, charges, eta=static.eta_ewald)
    return e_band - e_h + e_xc - e_dc + e_ii
