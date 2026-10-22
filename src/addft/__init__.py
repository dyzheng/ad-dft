import jax as _jax

_jax.config.update("jax_enable_x64", True)

"""ad-dft: 自动微分原生、periodic-first 的 DFT 内核。

分层（方案 §3.2）：
    L0 runtime     单位 / dtype / 精度账本
    L1 geometry    Cell / Basis(冻结壳层) / KMesh / Pseudo
    L2 physics     density / hartree / xc(energy-only) / local / nonlocal / ewald / occupation
    L3 solver      hamiltonian / eigh(子空间不变) / mixer / scf
    L4 diff        不动点隐式微分
    L5 observables 能量分解 / 力 / 应力 / (极化 / 约束)
"""
from .api import (  # noqa: F401
    System,
    energy,
    energy_and_forces,
    energy_explicit,
    energy_decomposition,
    make_energy,
    scf,
    setup,
    setup_from_case,
    stress,
)
from .geometry.cell import Cell, strained, volume  # noqa: F401
from .geometry.kmesh import gamma, monkhorst_pack  # noqa: F401
from .geometry.psp import Pseudo, synthetic  # noqa: F401
from .runtime import units  # noqa: F401

__all__ = [
    "System", "setup", "setup_from_case", "scf", "energy", "energy_explicit",
    "energy_and_forces", "stress",
    "energy_decomposition", "make_energy", "Cell", "strained", "volume",
    "gamma", "monkhorst_pack", "Pseudo", "synthetic", "units",
]
