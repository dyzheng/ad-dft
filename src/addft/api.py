"""对外 API（附录 D 草案的可执行版本）。

设计要点：
  1. `System` 把可微几何与静态离散化分开；对外的可微入口只有 `Cell`；
  2. 只有一个真正的可微标量 `energy(geo)`，其余观测量都是它的导数：
     forces = −grad(energy) 经隐式伴随，stress = grad_eps(energy)/Ω；
  3. `custom_vjp` 的 fwd 里跑完整 SCF，bwd 里只做一次伴随求解 ——
     显存与迭代数无关（这是「不展开 SCF」的全部意义）。
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .diff.implicit import forces_from_grad, implicit_grad, stress_from_grad
from .geometry.cell import Cell, strained, volume
from .geometry.kmesh import KMesh, gamma
from .geometry.psp import Pseudo, synthetic
from .observables.energy import energy_terms, energy_variational, total_energy
from .solver.scf import solve_scf
from .system import SCFState, Static, make_static


class System(NamedTuple):
    geo: Cell
    static: Static


def setup(lattice, frac, psp: Pseudo | None = None, ecut: float = 8.0,
          nband: int = 4, kmesh: KMesh | None = None, sigma: float = 0.05,
          density_factor: float = 4.0, eta_ewald: float = 0.9) -> System:
    geo = Cell(lattice=jnp.asarray(lattice, dtype=jnp.float64),
               frac=jnp.asarray(frac, dtype=jnp.float64))
    psp = synthetic() if psp is None else psp
    static = make_static(geo, psp, kmesh or gamma(), ecut, nband, sigma,
                         density_factor, eta_ewald)
    return System(geo=geo, static=static)


def scf(system: System, **kwargs) -> SCFState:
    return solve_scf(system.geo, system.static, **kwargs)


def energy(system: System, state: SCFState | None = None):
    """生产能量：变分形式（驻点 ⇒ 一阶导数无需响应项）。"""
    state = scf(system) if state is None else state
    return energy_variational(system.geo, system.static, state)


def energy_explicit(system: System, state: SCFState | None = None):
    """显式泛函形式（与 ABACUS fp_energy.cpp 的分解同构），用于能量分解与交叉验证。"""
    state = scf(system) if state is None else state
    return total_energy(system.geo, system.static, state)


def make_energy(static: Static, mode: str = "envelope", **scf_kwargs):
    """把「SCF + 能量」包成一个可微标量函数 f(geo)。

    这是整个软件唯一允许暴露给优化器/训练循环的物理入口。

    两条反向路径，**默认走包络定理**：

    * ``mode="envelope"``（默认）：在收敛点 ψ 冻结，直接取 ∂E/∂θ。
      成立条件是 E 是 KS 变分泛函、且 ψ 是 H[ρ*] 的本征矢（`solve_scf` 的抛光步骤
      保证 ρ* = F(ρ*)）。实测与重跑 SCF 的中心差分相对误差 ~1e-6，
      成本 = 一次能量评估，**不构造 Jacobian、不显存爆炸**。
    * ``mode="adjoint"``：完整不动点隐式微分 ``Aᵀλ = ∂L/∂n``。
      用于**非能量**观测量（DeltaQ 的 ∫w·n、极化、Born 电荷），
      那里包络定理不成立、λ ≠ 0。见 observables/constraint.py。

    两者的关系不是"前者是后者的近似"，而是"前者是后者在 ∂E/∂n = 0 时的退化"。
    """

    @jax.custom_vjp
    def _energy(geo):
        state = solve_scf(geo, static, **scf_kwargs)
        return energy_variational(geo, static, state)

    def _fwd(geo):
        state = solve_scf(geo, static, **scf_kwargs)
        e = energy_variational(geo, static, state)
        return e, (geo, state)

    def _bwd(res, cotangent):
        geo, state = res
        loss = lambda g_, st_: cotangent * energy_variational(g_, static, st_)
        if mode == "envelope":
            # 状态冻结 ⇒ 包络定理
            return (jax.grad(lambda gg: loss(gg, state))(geo),)
        return (implicit_grad(geo, static, state, lambda g_, s_, st_: loss(g_, st_)),)

    _energy.defvjp(_fwd, _bwd)
    return _energy


def energy_and_forces(system: System, **scf_kwargs):
    """E 与 F（eV/Å 由调用方换算；内部 Hartree/Bohr）。力来自隐式伴随，不是有限差分。"""
    e_fn = make_energy(system.static, **scf_kwargs)
    e = e_fn(system.geo)
    g = jax.grad(e_fn)(system.geo)
    return e, forces_from_grad(system.geo, g)


def stress(system: System, **scf_kwargs):
    """σ = (1/Ω) dE/dε；ε 作用于晶格，分数坐标固定（与 ABACUS 应力定义一致）。"""
    e_fn = make_energy(system.static, **scf_kwargs)

    def e_of_eps(eps):
        return e_fn(strained(system.geo, eps))

    g = jax.grad(e_of_eps)(jnp.zeros((3, 3)))
    return stress_from_grad(volume(system.geo.lattice), 0.5 * (g + g.T))


def energy_decomposition(system: System, state: SCFState | None = None, active=None):
    state = scf(system) if state is None else state
    kwargs = {} if active is None else {"active": active}
    return energy_terms(system.geo, system.static, state, **kwargs)
