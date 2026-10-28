"""L2 原语 · 占据数、费米能与 Mermin 自由能 (spec/occupation.md).

三件事必须在同一处定义，否则力会悄悄用错能量：
  1. f(ε) = 1/(1+e^{(ε-μ)/σ})，σ 即 k_B T（Hartree）；
  2. μ 由 Σ w f · g_s = N_e 隐式确定 —— 标量隐式微分，漏掉它电子数不守恒；
  3. 力必须来自自由能 F = E − σ·S̃，而不是 E。用 E 求力的误差 −T∂S/∂R 在
     σ = 0.1 eV 时可达几十 meV/Å，且符号系统，会让结构优化朝「熵增」方向走。

σ → 0 的硬约束：∂f/∂ε 的分母趋零，金属低温极限必须走整数占据分支，不能靠把 σ 调小。
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def fermi(e, mu, sigma):
    return jax.nn.sigmoid((mu - e) / sigma)


@partial(jax.jit, static_argnums=(3, 4))
def occupations(eig, weights, nelec, sigma, spin_degeneracy=2.0, iters=80):
    """返回 (f, mu)。二分求根（与 ABACUS Occupy::fermig 同口径）。

    ``sigma``/``spin_degeneracy`` 是静态参数：二分是 80 次固定迭代，
    在 eager 模式下会变成 80 次 dispatch（实测 ~60 ms）；jit 后整个循环
    在一次编译里执行（实测 <1 ms）。这两个量在 SCF 内本来就是常数。
    """
    if sigma < 1e-6:
        return _integer_occupation(eig, weights, nelec, spin_degeneracy), jnp.asarray(0.0)

    lo = jnp.min(eig) - 30.0 * sigma - 1.0
    hi = jnp.max(eig) + 30.0 * sigma + 1.0

    def body(_, bounds):
        a, b = bounds
        mid = 0.5 * (a + b)
        n = jnp.sum(weights[:, None] * fermi(eig, mid, sigma)) * spin_degeneracy
        too_many = n > nelec
        # 电子太多 ⇒ μ 偏高 ⇒ 收缩上界
        return (jnp.where(too_many, a, mid), jnp.where(too_many, mid, b))

    lo, hi = lax.fori_loop(0, iters, body, (lo, hi))
    mu = 0.5 * (lo + hi)
    return fermi(eig, mu, sigma), mu


def _integer_occupation(eig, weights, nelec, spin_degeneracy):
    """T = 0：按能量升序填，最后一个能级按需分数占据（简并时取平均）。"""
    neff = nelec / spin_degeneracy
    flat_e = eig.reshape(-1)
    flat_w = jnp.broadcast_to(weights[:, None], eig.shape).reshape(-1)
    order = jnp.argsort(flat_e)
    cum = jnp.cumsum(flat_w[order])
    filled = cum <= neff + 1e-12
    last = jnp.argmax(cum > neff - 1e-12)
    frac = jnp.clip((neff - (cum[last] - flat_w[order][last])) / flat_w[order][last], 0.0, 1.0)
    vals = jnp.where(filled, 1.0, 0.0)
    vals = vals.at[last].set(jnp.where(filled[last], vals[last], frac))
    out = jnp.zeros_like(flat_w).at[order].set(vals)
    return out.reshape(eig.shape)


def entropy(eig, occ, weights, sigma):
    """S̃ = −Σ w [f ln f + (1−f) ln(1−f)]（单位：k_B）。"""
    f = jnp.clip(occ, 1e-14, 1.0 - 1e-14)
    s = -(f * jnp.log(f) + (1.0 - f) * jnp.log(1.0 - f))
    return jnp.sum(weights[:, None] * s)


def free_energy(e_band, sigma, s_tilde):
    """Mermin 自由能。力必须对它求导。"""
    return e_band - sigma * s_tilde
