"""对外 API（附录 D 草案的可执行版本）。

设计要点：
  1. `System` 把可微几何与静态离散化分开；对外的可微入口只有 `Cell`；
  2. 能量观测量（力、应力）走**变分包络**：冻结 ψ，但密度按当前几何
     由 ρ(geo,ψ) 重建。位置与晶格应变共用同一条路径；
  3. 非能量观测量（ΔQ、极化、Born 电荷）走 `implicit_grad` 的
     `Aᵀλ=∂L/∂n` 伴随 —— 那里包络定理不成立；
  4. `custom_vjp` 的 fwd 里跑完整 SCF，bwd 只做一次能量梯度（或一次伴随
     求解），显存与 SCF 迭代数无关。
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .diff.implicit import forces_from_grad, implicit_grad, stress_from_grad
from .geometry.cell import Cell, strained, volume
from .geometry.kmesh import KMesh, gamma
from .geometry.psp import Pseudo, synthetic
from .observables.energy import energy_terms, mermin_free_energy, total_energy
from .physics.density import density_from_psi
from .solver.scf import solve_scf
from .system import SCFState, Static, make_static


class System(NamedTuple):
    geo: Cell
    static: Static


def setup(lattice, frac, psp: Pseudo | None = None, ecut: float = 8.0,
          nband: int = 4, kmesh: KMesh | None = None, sigma: float = 0.05,
          density_factor: float = 4.0, eta_ewald: float = 0.9,
          spec_idx=None) -> System:
    geo = Cell(lattice=jnp.asarray(lattice, dtype=jnp.float64),
               frac=jnp.asarray(frac, dtype=jnp.float64))
    psp = synthetic() if psp is None else psp
    static = make_static(geo, psp, kmesh or gamma(), ecut, nband,
                         spec_idx=spec_idx, sigma=sigma,
                         density_factor=density_factor, eta_ewald=eta_ewald)
    return System(geo=geo, static=static)


def setup_from_case(case_dir):
    """从 ABACUS 算例目录（INPUT/KPT/STRU/UPF）构造 System。"""
    from adapters.abacus.io import setup_from_case as _setup_from_case
    return _setup_from_case(case_dir)


def scf(system: System, **kwargs) -> SCFState:
    return solve_scf(system.geo, system.static, **kwargs)


def energy(system: System, state: SCFState | None = None):
    """生产能量：Mermin 自由能 F = E_var − σ·g_s·S̃（σ=0 时即总能）。"""
    state = scf(system) if state is None else state
    return mermin_free_energy(system.geo, system.static, state)


def energy_explicit(system: System, state: SCFState | None = None):
    """显式泛函形式（与 ABACUS fp_energy.cpp 的分解同构），用于能量分解与交叉验证。"""
    state = scf(system) if state is None else state
    return total_energy(system.geo, system.static, state)


def make_energy(static: Static, mode: str = "envelope",
                adjoint_mode: str = "matrix_free", **scf_kwargs):
    """把「SCF + 自由能」包成一个可微标量函数 f(geo)。

    这是整个软件唯一允许暴露给优化器/训练循环的物理入口。

    两条反向路径：

    * ``mode="envelope"``（默认，生产路径）：冻结收敛点的 ψ 与占据 f，
      但**密度按当前几何由冻结的 ψ 重建**：``ρ(geo,ψ)``。这一点是应力
      正确性的关键。若把实空间密度数组也冻结，晶格应变下会丢掉
      ``psi_to_real`` 的 G 缩放与 ``1/√Ω`` 因子，实测 Si/Γ 偏差数千 kbar；
      冻结 ψ 并让 ρ 跟随，包络定理对位置与晶格应变同时成立。
      占据冻结 + Mermin 自由能（∂F/∂f=0）保证有限温度下的占据响应也为零。
      成本 = 一次能量评估，**不构造 Jacobian、不需要响应求解**。
    * ``mode="adjoint"``：不动点隐式微分 ``Aᵀλ = ∂L/∂n``。这里假定 loss 只
      通过 ρ 依赖状态（ΔQ 的 ∫w·n、极化、Born 电荷）；对变分能量，还需补
      ``ψ`` 对 ρ 的隐函数响应，因此能量导数不走这条路径。见
      observables/constraint.py。
    """

    def _value(geo, state):
        return mermin_free_energy(geo, static, state)

    @jax.custom_vjp
    def _energy(geo):
        state = solve_scf(geo, static, **scf_kwargs)
        return _value(geo, state)

    def _fwd(geo):
        state = solve_scf(geo, static, **scf_kwargs)
        return _value(geo, state), (geo, state)

    def _bwd(res, cotangent):
        geo, state = res
        loss = lambda g_, st_: cotangent * _value(g_, st_)
        if mode == "envelope":
            def frozen_density_loss(g_):
                # ψ 冻结、ρ 跟随几何；S(f) 只依赖占据，故与 E 的梯度相同。
                rho = density_from_psi(static.basis, g_.lattice, state.psi,
                                       state.occ, static.kweights,
                                       static.spin_degeneracy)
                return loss(g_, state._replace(rho_r=rho))
            return (jax.grad(frozen_density_loss)(geo),)
        return (implicit_grad(geo, static, state,
                              lambda g_, s_, st_: loss(g_, st_),
                              mode=adjoint_mode),)

    _energy.defvjp(_fwd, _bwd)
    return _energy


def energy_and_forces(system: System, **scf_kwargs):
    """F 与力（eV/Å 由调用方换算；内部 Hartree/Bohr）。力来自变分包络，不是有限差分。"""
    e_fn = make_energy(system.static, **scf_kwargs)
    e = e_fn(system.geo)
    g = jax.grad(e_fn)(system.geo)
    return e, forces_from_grad(system.geo, g)


def stress(system: System, **scf_kwargs):
    """σ = −(1/Ω) dF/dε；ε 作用于晶格，分数坐标固定。

    符号与 ABACUS `TOTAL-STRESS (kbar)` 一致：压缩为正。内部导数
    ``dF/dε`` 与打印应力相差一个负号，二者不能混用。

    应变下必须冻结 ψ 而让 ``ρ(geo,ψ)`` 跟随（见 `make_energy`）：
    冻结实空间密度数组会丢掉 G 缩放/体积因子，Si/Γ 实测偏数千 kbar。
    """
    e_fn = make_energy(system.static, **scf_kwargs)

    def e_of_eps(eps):
        return e_fn(strained(system.geo, eps))

    g = jax.grad(e_of_eps)(jnp.zeros((3, 3)))
    return stress_from_grad(volume(system.geo.lattice), 0.5 * (g + g.T))


def energy_decomposition(system: System, state: SCFState | None = None, active=None):
    state = scf(system) if state is None else state
    kwargs = {} if active is None else {"active": active}
    return energy_terms(system.geo, system.static, state, **kwargs)
