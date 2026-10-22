"""原语层测试（L0/L1）：解析极限 + 伴随恒等式 + 表示一致性。

约定：**已知失败用 xfail 标注并写明实测数字**，不用放宽容差的方式"通过"。
这是校验方案的一部分——失败项本身就是任务卡（见 docs/TASKS.md）。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import addft as ad
from addft.geometry.basis import density_g, gvectors, real_from_g, psi_to_real
from addft.geometry.cell import volume
from addft.physics import hartree, nonlocal_op, xc
from addft.physics.ewald import ewald_energy
from addft.solver.scf import vmat_from_real
from conftest import LATTICE, make_system


def test_hartree_fft_vs_explicit_sum(system):
    """I-2（L0 版）：FFT 路径 vs 显式 G 求和（ref 用最笨的写法）。

    Hartree 只做倒空间乘法，因此两个实现必须到机器精度一致——
    这一条抓的是 FFT 归一化（1/N、1/Ω）与 G=0 处置。
    """
    basis = system.static.basis
    st = ad.scf(system)
    om = volume(LATTICE)
    ng = density_g(basis, LATTICE, st.rho_r)
    q2 = jnp.sum(gvectors(basis, LATTICE) ** 2, axis=-1)

    vg = jnp.where(q2 > 1e-14, 4.0 * jnp.pi * ng / jnp.where(q2 > 1e-14, q2, 1.0), 0.0)
    v_fft = hartree.hartree_potential_r(basis, LATTICE, st.rho_r)

    # ref：逐点显式求和
    g = gvectors(basis, LATTICE)
    shape = tuple(basis.fft_shape)
    grid = jnp.stack(jnp.meshgrid(*(jnp.arange(n) for n in shape), indexing="ij"), axis=-1)
    r = (grid.reshape(-1, 3) / jnp.array(shape, dtype=jnp.float64)) @ LATTICE
    phase = jnp.exp(1j * (r @ g.T))  # (nfft, nG)
    v_ref = jnp.real(phase @ vg)

    assert float(jnp.max(jnp.abs(v_fft - v_ref))) < 1e-10


def test_hartree_planewave_exact(system):
    """I-2：对**带限**密度，Hartree 势有闭式解，必须逐点精确。

    取 n(r) = 1 + Σ_i a_i cos(G_i·r)（G_i 在球内），则
        V_H(r) = Σ_i 4π a_i/|G_i|² cos(G_i·r)   （G=0 项按契约置零）

    （推导：n_G = a/2 ⇒ V_G = 4πn_G/G² = 2πa/G²；V(r) = V_G e^{iGr} + c.c. = 4πa/G² cos）
    这个 oracle 没有截断误差，因此判据是机器精度——它能抓 FFT 归一化、G=0 处置、
    以及"复数中间量被静默转成实数"这类错误。
    """
    from conftest import LATTICE

    basis = system.static.basis
    g = gvectors(basis, LATTICE)
    shape = tuple(basis.fft_shape)
    grid = jnp.stack(jnp.meshgrid(*(jnp.arange(n) for n in shape), indexing="ij"), axis=-1)
    r = (grid.reshape(-1, 3) / jnp.array(shape, dtype=jnp.float64)) @ LATTICE

    amps = [0.7, -0.4, 0.25]
    idxs = [1, 4, 10]
    rho = jnp.ones(basis.nfft)
    exact = jnp.zeros(basis.nfft)
    for a, i in zip(amps, idxs):
        g2i = float(jnp.sum(g[i] * g[i]))
        rho = rho + a * jnp.cos(r @ g[i])
        exact = exact + 4.0 * jnp.pi * a / g2i * jnp.cos(r @ g[i])

    v = hartree.hartree_potential_r(basis, LATTICE, rho)
    assert float(jnp.max(jnp.abs(v - exact))) < 1e-10


def test_hartree_exact_linearity(system):
    """v(ρ+δ) − v(ρ) 必须严格等于 v(δ)（线性算子的存在性检验）。

    这一条抓的是"算子被写成了非线性"或"中间表示被静默转换"。
    实测案例：`hartree_potential_r` 里一句 `vg.astype(rho_r.dtype)` 把 complex128
    转成 float64，丢掉 n_G 的虚部 —— 破坏 n(-G)=n(G)*，势等价于"密度先对称化"。
    SCF 照样收敛、能量只差几 mHa，而线性性偏差 21%，梯度全错。
    """
    basis = system.static.basis
    st = ad.scf(system)
    d = jnp.sin(jnp.arange(basis.nfft) * 0.71)
    d = d - jnp.mean(d)
    d = d / jnp.linalg.norm(d)

    lhs = (hartree.hartree_potential_r(basis, LATTICE, st.rho_r + d)
           - hartree.hartree_potential_r(basis, LATTICE, st.rho_r))
    rhs = hartree.hartree_potential_r(basis, LATTICE, d)
    assert float(jnp.max(jnp.abs(lhs - rhs))) < 1e-10  # 见 K12 失败记录


def test_grad_div_adjoint(system):
    """I-9：⟨∇f, v⟩ = −⟨f, ∇·v⟩ 到机器精度（GGA 的 −∇· 项依赖这条）。"""
    basis = system.static.basis
    om, n = volume(LATTICE), basis.nfft
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    f = jax.random.normal(k1, (n,))
    v = jax.random.normal(k2, (3, n))
    lhs = (om / n) * jnp.sum(xc.grad_r(basis, LATTICE, f) * v)
    rhs = -(om / n) * jnp.sum(f * xc.div_r(basis, LATTICE, v))
    assert abs(float(lhs - rhs)) < 1e-10 * max(abs(float(lhs)), 1.0)


def test_xc_potential_matches_pointwise_oracle(system):
    """v_xc 必须逐点等于 d(n·ε)/dn —— 这是 `N/Ω` 权重约定的唯一判据。

    用解析点级 oracle（不是差分），因此判据是机器精度：
    漏掉 (Ω/N) 会让势整体小 N/Ω 倍，而总能只差几 mHa。
    """
    basis = system.static.basis
    st = ad.scf(system)
    v = xc.xc_potential_r(xc.xc_energy_lda, basis, LATTICE, st.rho_r)
    v_exact = jax.vmap(xc.xc_potential_point)(st.rho_r)
    assert float(jnp.max(jnp.abs(v - v_exact))) < 1e-10


def test_ewald_eta_independence(system):
    """I-4：Ewald 对 η 无关（实空间/倒空间/自能/背景四项必须成对出现）。"""
    z = system.static.psp.zval[system.static.spec_idx]
    vals = [float(ewald_energy(system.geo, z, eta=e, nmax=3)) for e in (0.9, 1.0, 1.1)]
    assert max(vals) - min(vals) < 2e-4, f"Ewald η 相关性过大: {vals}"


def test_ewald_translation_invariant(system):
    from addft.geometry.cell import with_frac

    z = system.static.psp.zval[system.static.spec_idx]
    e0 = ewald_energy(system.geo, z)
    e1 = ewald_energy(with_frac(system.geo, system.geo.frac + jnp.array([0.03, 0.07, 0.11])), z)
    assert abs(float(e0 - e1)) < 1e-12


def test_matrix_vs_grid_representation(system):
    basis = system.static.basis
    st = ad.scf(system)
    om, n = volume(LATTICE), basis.nfft
    vg = density_g(basis, LATTICE, st.rho_r)
    v = real_from_g(basis, LATTICE, vg)
    psi = st.psi[0][0]
    psi_r = psi_to_real(basis, LATTICE, psi)
    mat = jnp.real(jnp.vdot(psi, vmat_from_real(basis, v) @ psi))
    integ = (om / n) * jnp.sum(v * jnp.abs(psi_r) ** 2)
    assert abs(float(mat - integ)) < 1e-10


def test_nonlocal_operator_is_hermitian(system):
    basis = system.static.basis
    st = ad.scf(system)
    proj = nonlocal_op.projectors(system.static.psp, system.static.spec_idx,
                                  basis, system.geo, system.static.kpts)
    d = nonlocal_op.dion_full(system.static.psp, system.static.spec_idx)
    x = jax.random.normal(jax.random.PRNGKey(3), st.psi[0].shape) + 0j
    y = jax.random.normal(jax.random.PRNGKey(4), st.psi[0].shape) + 0j
    ax = nonlocal_op.vnl_apply(proj, d, x[None])
    ay = nonlocal_op.vnl_apply(proj, d, y[None])
    lhs = jnp.sum(jnp.conj(y[None]) * ax)
    rhs = jnp.sum(jnp.conj(ay) * x[None])
    assert abs(complex(lhs - rhs)) < 1e-10 * max(abs(complex(lhs)), 1.0)


def test_dion_packing_is_block_diagonal():
    """I-9 的构造性前提：D 必须是按原子分块的块对角矩阵。

    旧实现 (nat,nat,npj,npj).reshape(...) 会把 D[0,1] 块摊到全局
    row 0 / col 3，导致算子不再 Hermitian；这个测试直接锁住布局。
    """
    from addft.physics.nonlocal_op import dion_full
    from addft.geometry.psp import synthetic

    psp = synthetic()
    si = jnp.array([0, 0], dtype=jnp.int32)
    d = dion_full(psp, si)
    assert d.shape == (4, 4)
    assert float(jnp.max(jnp.abs(d - jnp.conj(d).T))) < 1e-14
    assert float(jnp.max(jnp.abs(d[2:, :2]))) < 1e-14
    assert float(jnp.max(jnp.abs(d[:2, 2:]))) < 1e-14
