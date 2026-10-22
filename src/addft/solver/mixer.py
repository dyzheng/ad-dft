"""L3 混合器（加速器，**永不进入微分图**）。

为什么混合器可以、而且必须被排除在反向传播之外（方案 §4.7 的精确结论）：

    线性混合 M(n) = (1−α)n + αF(n)：在不动点，I − ∂M = α(I − ∂F)，∂_θ M = α ∂_θ F，
    α 在伴随方程里被约掉，因此**固定 α 不影响一阶导数**。
    Broyden/DIIS 的系数由历史残差决定，α 是 (n, 历史) 的函数；在精确不动点第三项
    ∂α/∂n·(F(n) − n) 消失，但实际残差 10⁻⁶ 时这一项为 (∂α/∂n)×10⁻⁶，无法界定。

所以旁路混合器不是数学上的必需，而是**把「梯度是否可靠」与「混合器调得好不好」解耦**
的工程选择。这也是本文件只提供 value-level API、不提供任何 jax 变换的原因。
"""

from __future__ import annotations

from typing import List

import jax.numpy as jnp


class DIISMixer:
    """Pulay/DIIS 混合（与 ABACUS pulay_mixing.cpp 同一族）。

    历史长度 m 固定，避免动态 shape；残差爆炸时自动清空历史（restart）。
    """

    def __init__(self, size: int, m: int = 8, beta: float = 0.5):
        self.size = int(size)
        self.m = int(m)
        self.beta = float(beta)
        self._rins: List[jnp.ndarray] = []
        self._routs: List[jnp.ndarray] = []

    def reset(self) -> None:
        self._rins.clear()
        self._routs.clear()

    def __call__(self, rho_in, rho_out):
        self._rins.append(rho_in)
        self._routs.append(rho_out)
        if len(self._rins) > self.m:
            self._rins.pop(0)
            self._routs.pop(0)

        if len(self._rins) == 1:
            return rho_in + self.beta * (rho_out - rho_in)

        n = len(self._rins)
        res = [self._routs[i] - self._rins[i] for i in range(n)]
        B = jnp.zeros((n + 1, n + 1), dtype=rho_in.dtype)
        for i in range(n):
            for j in range(n):
                B = B.at[i, j].set(jnp.dot(res[i], res[j]))
        B = B.at[:n, n].set(-1.0).at[n, :n].set(-1.0)
        rhs = jnp.zeros((n + 1,), dtype=rho_in.dtype).at[n].set(-1.0)
        # 轻微正则化，避免早期残差共线导致奇异
        B = B + jnp.eye(n + 1, dtype=rho_in.dtype) * 1e-12
        try:
            c = jnp.linalg.solve(B, rhs)[:n]
        except Exception:  # pragma: no cover
            self.reset()
            return rho_in + self.beta * (rho_out - rho_in)

        c = jnp.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        new = jnp.zeros_like(rho_in)
        for i in range(n):
            new = new + c[i] * (self._rins[i] + self.beta * res[i])
        if not bool(jnp.all(jnp.isfinite(new))):
            self.reset()
            return rho_in + self.beta * (rho_out - rho_in)
        return new
