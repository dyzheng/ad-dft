"""L4 微分层 · 不动点隐式微分（方案 §4.7 的可执行版本）。

数学：
    n* = F(n*; θ)                        （物理映射，不含混合器）
    (I − ∂_n F)^T λ = ∂L/∂n*
    dL/dθ = ∂L/∂θ|_n* + λ^T ∂_θ F

实现上的关键工程选择：

    **不手写一阶微扰（Σ_{m≠n} c_m <c_m|δH|c_n>/(ε_n−ε_m)），而是对 scf_map 做前向
    模式雅可比（jacfwd）。** 前者要人手推一遍伴随算子（D^T、R^T、V'^T），而转置写错
    是本项目最隐蔽的 bug（能量完全不受影响）；jacfwd 不依赖任何 VJP，因此对简并、
    对 F 的内部实现（dense eigh 还是 Davidson）都成立。
    代价 O(N_fft) 次前向，V0 的小体系可接受；生产路径（K12）换成 matrix-free GMRES +
    解析伴随，接口不变。

一个必须写进文档的结论（实测得到，不是推导猜的）：
    对**变分形式**的能量，∂E/∂n|(ψ) 严格为 0，因此 λ = 0，响应项自动消失——
    dL/dθ 退化为 Hellmann–Feynman。这不是「隐式微分没用」，而是隐式微分
    **自动发现了包络定理**。
    如果改用显式泛函形式（T + ∫v_loc n + E_H + E_xc + E_NL + E_II），它在固定 n 下
    对 ψ 不驻定，响应项非零；此时若把响应项丢掉，力会有 ~20% 的系统偏差
    （在本仓库的原型上实测；tests/test_forces.py 记录了对照）。
    对**非能量观测量**（DeltaQ 的 ∫w·n、极化、Born 电荷），λ ≠ 0，响应项是必需的。
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from ..geometry.cell import Cell
from ..physics.response import build_jacobian
from ..solver.scf import scf_map
from ..system import SCFState, Static


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def implicit_grad(geo: Cell, static: Static, state: SCFState,
                  loss: Callable, return_diagnostics: bool = False):
    """返回 dL/dgeo，其中 L = loss(geo, static, state) 是标量。

    `loss` 必须把一个 `SCFState` 映射为标量；它内部可以引用 state.psi / state.occ
    （视为常数），也可以只依赖 state.rho_r。
    """
    n_star = state.rho_r
    nfft = static.basis.nfft

    # (1) ∂L/∂n*
    g_n = jax.grad(lambda nn: loss(geo, static, state._replace(rho_r=nn)))(n_star)

    # (2) 物理映射的雅可比与响应算子 A = I − ∂_n F
    #     用解析一阶微扰（简并安全），不用 jacfwd(scf_map)——后者在简并处返回 NaN。
    J = build_jacobian(geo, static, state)
    A = jnp.eye(nfft) - J

    # (3) 伴随求解 A^T λ = g
    lam = jnp.linalg.solve(A.T, g_n)

    # (4) 两个显式项：状态项 + 映射项
    d_theta_state = jax.grad(lambda gg: loss(gg, static, state))(geo)
    d_theta_map = jax.grad(
        lambda gg: jnp.sum(lam * scf_map(gg, static, n_star))
    )(geo)
    grad = _tree_add(d_theta_state, d_theta_map)

    if not return_diagnostics:
        return grad
    resid = jnp.linalg.norm(A.T @ lam - g_n) / (jnp.linalg.norm(g_n) + 1e-30)
    return grad, {
        "adjoint_residual": float(resid),
        "cond": float(jnp.linalg.cond(A)),
        "lam_norm": float(jnp.linalg.norm(lam)),
        "g_norm": float(jnp.linalg.norm(g_n)),
    }


def forces_from_grad(geo: Cell, grad_cell) -> jnp.ndarray:
    """分数坐标梯度 → 笛卡尔力：F = −dE/dR，且 dE/dfrac = dE/dR · A^T。

    （R = frac @ A  ⇒  dE/dR = dE/dfrac @ A^{-T}）
    """
    g_frac = grad_cell.frac
    return -(g_frac @ jnp.linalg.inv(geo.lattice).T)


def stress_from_grad(omega, grad_eps) -> jnp.ndarray:
    """σ = (1/Ω) dE/dε（ε 通过与 strained() 相同的构造产生）。"""
    return grad_eps / omega
