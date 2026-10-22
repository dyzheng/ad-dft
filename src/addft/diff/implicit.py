"""L4 微分层 · 不动点隐式微分（方案 §4.7 的可执行版本）。

数学：
    n* = F(n*; θ)                        （物理映射，不含混合器）
    (I − ∂_n F)^T λ = ∂L/∂n*
    dL/dθ = ∂L/∂θ|_n* + λ^T ∂_θ F

生产路径使用 matrix-free 转置：`make_response` 只提供纯线性 apply，
`jax.linear_transpose` 给出 A^T 的 matvec，再用 BiCGSTAB 解 A^T λ = g。
稠密 Jacobian 只保留给小体系诊断（nfft ≤ 1500）。
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab

from ..geometry.cell import Cell
from ..physics.response import build_jacobian, make_response
from ..solver.scf import scf_map
from ..system import SCFState, Static


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def implicit_grad(geo: Cell, static: Static, state: SCFState,
                  loss: Callable, return_diagnostics: bool = False,
                  mode: str = "matrix_free", rtol: float = 1e-10):
    """返回 dL/dgeo，其中 L = loss(geo, static, state) 是标量。

    `loss` 必须把一个 `SCFState` 映射为标量；它内部可以引用 state.psi / state.occ
    （视为常数），也可以只依赖 state.rho_r。
    """
    n_star = state.rho_r

    # (1) ∂L/∂n*
    g_n = jax.grad(lambda nn: loss(geo, static, state._replace(rho_r=nn)))(n_star)

    # (2) A = I − ∂_n F；dense 仅诊断，matrix_free 走转置 matvec
    if mode == "dense":
        J = build_jacobian(geo, static, state)
        A = jnp.eye(static.basis.nfft) - J
        lam = jnp.linalg.solve(A.T, g_n)
        aT = lambda y: A.T @ y
        info = 0
    else:
        apply = make_response(geo, static, state)
        apply_t = jax.linear_transpose(apply, n_star)

        def aT(y):
            return y - apply_t(y)[0]

        lam, _ = bicgstab(aT, g_n, tol=rtol, atol=0.0, maxiter=200)
        _res = jnp.linalg.norm(aT(lam) - g_n) / (
            jnp.linalg.norm(g_n) + 1e-30)
        info = 0 if float(_res) <= 1e-6 else 1
        if info != 0:
            raise RuntimeError(
                f"adjoint BiCGSTAB 未收敛: residual={float(_res):.3e}")

    # (3) 两个显式项：状态项 + 映射项
    d_theta_state = jax.grad(lambda gg: loss(gg, static, state))(geo)
    d_theta_map = jax.grad(
        lambda gg: jnp.sum(lam * scf_map(gg, static, n_star))
    )(geo)
    grad = _tree_add(d_theta_state, d_theta_map)

    if not return_diagnostics:
        return grad
    resid = jnp.linalg.norm(aT(lam) - g_n) / (jnp.linalg.norm(g_n) + 1e-30)
    return grad, {
        "adjoint_residual": float(resid),
        "solver_info": int(info),
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
