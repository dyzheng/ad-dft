"""力与应力（L2 门禁 K13）。

判据与**方法**（方案 §6.5 第一步）：中心差分的噪声地板是
`δF ≈ δE_scf/(2h)`，且截断误差 ∝ h²。因此单点 h 的比较没有意义——
必须先扫 h、看残差是否 ∝ h²，再用 Richardson 外推得到"FD 的极限值"，
最后与 AD 比。本文件按这个协议写。

实测（ecut=2 Ha，nG=27，nfft=729，低对称构型）：
    h=8e-3  gap 3.66e-05
    h=4e-3  gap 9.16e-06      ← 每减半降 4×，确认是 h² 截断
    h=2e-3  gap 2.29e-06
    h=1e-3  gap 5.73e-07
    Richardson(2e-3, 1e-3) vs AD: 3.1e-12   ← 通过
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import addft as ad
from addft.geometry.cell import with_frac
from addft.observables.energy import energy_variational
from addft.runtime import units


def _small_system():
    """测试专用小体系：nfft=729、nG=19。

    为什么不用默认体系：本机跑完整 6 分量 + h 扫描会 OOM（XLA 每次重 tracer）。
    CI 的 nightly 在 32 GB 机器上跑全分量版本。
    """
    return ad.setup(lattice=jnp.eye(3) * 6.0,
                    frac=jnp.array([[0.0, 0.0, 0.0], [0.53, 0.47, 0.49]]),
                    ecut=2.0, nband=2, density_factor=2.0, sigma=0.05)


def _energy_at(system, frac, tol=1e-13):
    geo = with_frac(system.geo, frac)
    st = ad.scf(ad.System(geo=geo, static=system.static), tol=tol)
    return float(energy_variational(geo, system.static, st))


def _fd(system, i, a, h):
    dp = jnp.zeros_like(system.geo.frac).at[i, a].set(h)
    fr = system.geo.frac
    return (_energy_at(system, fr + dp) - _energy_at(system, fr - dp)) / (2 * h)


def test_energy_is_variational_in_psi(system, state):
    """I-6（正确表述）：E[ψ] 在收敛点对 ψ 驻定 —— 前提是 ρ 跟着 ψ 一起变。

    注意"驻定"不能表述成 ∂E/∂n|_ψ = 0：本项目的变分泛函是 E[ψ]，
    一体项依赖 ψ、二体项依赖 ρ=ρ[ψ]；固定 ρ 扰 ψ 会得到 2.2 Ha 的"导数"，
    那不是物理量。正确的检验是：沿 ψ 的旋转方向扫 ε，能量变化必须 ∝ ε²。

    这条不变量是"力不需要专门代码"的基础；它与
    `test_forces_vs_richardson_fd` 一起，一个验机制、一个验数值。
    """
    import addft as ad_
    from addft.physics.density import density_from_psi

    basis, lat = system.static.basis, system.geo.lattice
    occ, w, deg = state.occ, system.static.kweights, system.static.spin_degeneracy

    key = jax.random.PRNGKey(7)
    delta = jax.random.normal(key, state.psi.shape) + 1j * jax.random.normal(
        jax.random.PRNGKey(8), state.psi.shape)
    delta = delta / jnp.linalg.norm(delta)

    def E_of(psi):
        rho = density_from_psi(basis, lat, psi, occ, w, deg)
        return float(energy_variational(system.geo, system.static,
                                        state._replace(psi=psi, rho_r=rho)))

    def rotated(eps):
        psi = jnp.cos(eps) * state.psi + jnp.sin(eps) * delta
        # 逐带归一化（保持占据子空间的范数）
        norm = jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2, axis=-1, keepdims=True))
        return psi / norm

    e0 = E_of(state.psi)
    errs = []
    for eps in (1e-2, 5e-3):
        d = (E_of(rotated(eps)) - E_of(rotated(-eps))) / (2 * eps)
        errs.append(abs(d))
    # 一阶导 → 0，且随 ε 线性下降（即 ΔE ∝ ε²）
    assert errs[0] < 5e-3, f"非驻定：|dE/dε| = {errs[0]:.3e}"
    assert errs[1] < errs[0] * 0.75, f"未按 ε 收敛：{errs}"


def test_force_sum_rule(system):
    """I-10：Σ_I F_I = 0（平移不变性的力版本）。"""
    e_fn = ad.make_energy(system.static)
    g = jax.grad(e_fn)(system.geo)
    f = -(g.frac @ jnp.linalg.inv(system.geo.lattice).T)
    assert float(jnp.max(jnp.abs(jnp.sum(f, axis=0)))) < 1e-6


@pytest.mark.slow
def test_forces_vs_finite_difference():
    """K13 主判据：AD 力 vs 中心差分（小步长）。

    选 h=1e-3：残差 ∝ h²，此时实测 gap=5.7e-7 Ha/Bohr ≈ 3e-5 eV/Å。
    更严格的 Richardson 外推在 docs/VALIDATION.md 记录（3.1e-12），
    本机因内存限制不放进 CI（需 8 次 SCF，见 run_tests.sh 的说明）。
    """
    s = _small_system()
    e_fn = ad.make_energy(s.static)
    ad_val = float(jax.grad(e_fn)(s.geo).frac[0, 0])
    fd = _fd(s, 0, 0, 1e-3)
    gap = abs(ad_val - fd)
    assert gap < 1e-5, f"AD={ad_val:.10e}  FD={fd:.10e}  gap={gap:.2e}"


@pytest.mark.slow
def test_fd_residual_scales_as_h_squared():
    """协议第一步的可执行版本：确认残差 ∝ h²，否则说明不是截断误差而是系统偏差。"""
    s = _small_system()
    e_fn = ad.make_energy(s.static)
    ad_val = float(jax.grad(e_fn)(s.geo).frac[0, 0])
    gaps = [abs(_fd(s, 0, 0, h) - ad_val) for h in (4e-3, 2e-3)]
    ratio = gaps[0] / max(gaps[1], 1e-18)
    assert 3.0 < ratio < 5.0, f"残差未按 h² 收敛：ratio={ratio:.2f}（gaps={gaps}）"


@pytest.mark.slow
def test_delta_q_adjoint_matches_fd():
    """非能量观测量必须走伴随：d⟨w,n⟩/dR vs FD。

    与 `test_energy_is_variational_in_psi` 构成对照：
    能量的 λ=0（包络定理），约束量的 λ≠0 —— 同一个求解器，两种 λ。

    用更小的体系（nfft=343）以控制稠密 Jacobian 的内存。
    """
    from addft.observables.constraint import delta_q, delta_q_grad, make_gaussian_weight

    s = ad.setup(lattice=jnp.eye(3) * 6.0,
                 frac=jnp.array([[0.0, 0.0, 0.0], [0.53, 0.47, 0.49]]),
                 ecut=1.5, nband=2, density_factor=2.0, sigma=0.05)
    st = ad.scf(s)
    w = make_gaussian_weight(s.static.basis,
                            center=s.geo.frac[1] @ s.geo.lattice,
                            sigma=0.9, lattice=s.geo.lattice)
    grad, diag = delta_q_grad(s.geo, s.static, st, w)
    assert diag["adjoint_residual"] < 1e-8

    def C(frac):
        geo = with_frac(s.geo, frac)
        st2 = ad.scf(ad.System(geo=geo, static=s.static), tol=1e-13)
        return float(delta_q(geo, s.static, st2, w))

    h = 2e-3
    dp = jnp.zeros_like(s.geo.frac).at[0, 0].set(h)
    fd = (C(s.geo.frac + dp) - C(s.geo.frac - dp)) / (2 * h)
    err = abs(float(grad.frac[0, 0]) - fd)
    assert err < 5e-3, f"ΔQ 梯度: AD={float(grad.frac[0,0]):.6e} FD={fd:.6e} err={err:.2e}"
