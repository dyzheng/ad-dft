"""L0 运行时 · dtype 策略 (spec/discretization.md §dtype).

规则（写进契约，不按调用点临时决定）：

* 波函数系数 `psi`      : complex64（默认）/ complex128（参考）
* 密度、势、能量累加     : float64   —— 密度是二次型，降精度会污染 XC 与残差
* 赝势投影 `beta`        : float64
* 本征值                 : float64

`dtype.wavefunction` 一旦降到 complex64，性能报告必须同时报告能量漂移，
否则该加速比按 0 计（第 5 章反作弊条款 1）。这里把「漂移报告」做成
一个显式开关 `AccuracyLedger`，让「降精度」这个动作无法静默发生。
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

FLOAT = jnp.float64
COMPLEX = jnp.complex128
COMPLEX_LOWMEM = jnp.complex64


class AccuracyLedger:
    """记录所有「为了速度而放宽精度」的动作，性能 harness 强制读取。"""

    _entries: list[tuple[str, float]] = []

    @classmethod
    def record(cls, what: str, drift_ha: float) -> None:
        cls._entries.append((what, float(drift_ha)))

    @classmethod
    def report(cls) -> dict[str, float]:
        return dict(cls._entries)

    @classmethod
    def reset(cls) -> None:
        cls._entries.clear()
