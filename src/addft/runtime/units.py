"""L0 运行时 · 单位契约 (spec/units.md).

内部统一 Hartree / Bohr（`e^2 = 1`, `m_e = 1`, `hbar = 1`）。
ABACUS 内部统一 Rydberg / Bohr（`e^2 = 2`, 动能算符无 1/2 因子）。
两者的桥只允许出现在 adapters/abacus 与 IO 边界，**不允许**出现在 L1 以上。

任何跨层传递裸 float 而不带单位标签的函数，都在 CI 的 contract-check 里报错。
"""

from __future__ import annotations

HA_TO_EV = 27.211386245988
RY_TO_HA = 0.5
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HA_TO_KBAR = 2.942105e5  # 1 Ha/Bohr^3 = 294210.5 kbar (= 29421 GPa)


def ry_to_ha(x):
    """ABACUS ecutwfc / smearing_sigma (Ry) -> Hartree。"""
    return x * RY_TO_HA


def ha_to_ry(x):
    return x / RY_TO_HA


def ha_to_ev(x):
    return x * HA_TO_EV


def stress_ha_bohr3_to_kbar(s):
    return s * HA_TO_KBAR
