"""L3 求解器 · 本征问题 —— **只对子空间不变的量求导**（方案 §4.5）。

本文件里有一条硬规则，用代码而不是文档来保证：

    微分层只接受「子空间不变」的输出：总能量、密度矩阵 P = Σ f_n c_n c_n†、投影算子。
    任何对单个本征值的求导请求，必须在调用方被显式拒绝。

`forbid_eigenvalue_grad` 是那条规则的执行器：它把「对本征值求导」变成一个
**立刻抛异常**的操作，而不是在简并处悄悄返回 NaN。这样「梯度是 NaN」永远
不会被误读成「体系病态」。

生产路径（K09）用 Davidson/LOBPCG + 子空间 Rayleigh–Ritz；此处 dense eigh
是 ref 实现，也是 V0 在 nG < 2000 时的生产实现。
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

try:
    from ..kernels.eigh import load_eigh_lowest
    _NATIVE_EIGH = load_eigh_lowest()
except Exception:  # pragma: no cover - 取决于本地是否构建了 native/
    _NATIVE_EIGH = None


def eigh_lowest(H, nband: int):
    """H: (..., nG, nG) Hermitian → (eig (..., nband), vec (..., nband, nG))。

    有原生 cuSOLVER kernel 时走原生路径；否则回退到 `jnp.linalg.eigh`。
    原生路径只用于正向求解，微分层由 `scf_map` 的解析微扰提供，不穿过 eigh。
    """
    if _NATIVE_EIGH is not None and H.ndim == 3 and H.dtype == jnp.complex128:
        return _NATIVE_EIGH(H, int(nband))
    w, v = jnp.linalg.eigh(H)
    w = w[..., :nband]
    v = jnp.swapaxes(v[..., :, :nband], -1, -2)
    return w, v


def density_matrix(vec, occ):
    """P = Σ_n f_n c_n c_n† —— 子空间不变，简并安全。"""
    w = occ[..., :, None, None]
    return jnp.sum(w * vec[..., :, :, None] * jnp.conj(vec[..., :, None, :]), axis=-3)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def forbidden(fn_name: str, eigvals):
    return eigvals


def _forbidden_fwd(fn_name, eigvals):
    return eigvals, None


def _forbidden_bwd(fn_name, _, g):
    raise RuntimeError(
        f"禁止对单个本征值求导（来自 {fn_name}）。"
        "简并/近简并时 1/(ε_n − ε_m) 发散；请改用密度矩阵、投影算子或总能量。"
        "见 spec/invariants.md §subspace。"
    )


forbidden.defvjp(_forbidden_fwd, _forbidden_bwd)


def band_energy_diagnostic(eig, occ, weights, spin_degeneracy=2.0):
    """能带和（后处理/诊断用，**不进入力的路径**）。"""
    return jnp.sum(weights[:, None] * occ * eig) * spin_degeneracy
