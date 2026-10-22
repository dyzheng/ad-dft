"""七因子性能 harness（K17 骨架）。

原则：每个因子有**独立**的 harness、baseline 与记分；不许跨因子"借"收益。
任何 > 10% 的加速必须能归因到某个因子，否则视为测量噪声（反作弊第 6 条）。

因子（见 docs/VALIDATION.md 与方案 §5.2）：
  1 SCF 迭代数（32→13）          代理指标：迭代数分布
  2 本征求解重构                  代理指标：matvec 计数
  3 内核融合                      代理指标：global 往返次数
  4 混合精度                      必须附能量漂移（AccuracyLedger）
  5 零主机往返                    代理指标：同步点/迭代
  6 通信                          代理指标：通信占比
  7 内存布局                      代理指标：有效带宽

本文件只实现"能被测量的东西"：代理指标采集 + 反作弊检查 + 因子归因表。
真实 kernel 尚未实现（K20），因此数值均为占位。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict

import jax
import jax.numpy as jnp

import addft as ad
from addft.runtime.dtypes import AccuracyLedger


@dataclass
class FactorReport:
    factor: str
    metric: str
    value: float
    budget: float
    unit: str = ""
    passed: bool = False

    def __post_init__(self):
        self.passed = self.value <= self.budget


@dataclass
class RunReport:
    case: str
    inputs: dict = field(default_factory=dict)
    factors: list = field(default_factory=list)
    wall_clock_s: float = 0.0
    accuracy: dict = field(default_factory=dict)
    ledger: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def measure_scf_iterations(case_kwargs) -> float:
    """因子 1 的代理指标：固定体系下的 SCF 迭代数。"""
    s = ad.setup(**case_kwargs)
    st = ad.scf(s)
    return float(st.n_iter)


def measure_matvec_count(case_kwargs) -> float:
    """因子 2 的代理指标：本征求解每次调用的 matvec 数。

    V0 用稠密 eigh，matvec 数 = nG（一次完整对角化）。K09 的 Davidson 应显著降低。
    """
    s = ad.setup(**case_kwargs)
    return float(s.static.basis.nG)


def measure_global_roundtrips(case_kwargs) -> float:
    """因子 3 的代理指标：一次 scf_map 内的 global 级算子数（FFT + 矩阵乘）。"""
    from addft.solver.scf import scf_map

    s = ad.setup(**case_kwargs)
    st = ad.scf(s)
    # 代理指标：一次 scf_map 中"H 的构建次数 × nG²"（真实实现里换成 nsys 的
    # global load/store 计数与 kernel 启动数，这里用可复现的解析代理）
    jit = jax.jit(lambda rho: scf_map(s.geo, s.static, rho))
    jit(st.rho_r).block_until_ready()
    _ = jit.lower(st.rho_r).compile()   # 确认可编译；真实计数在 K20 接 nsys
    return float(s.static.basis.nG ** 2)


def check_anticheat(report: RunReport) -> list[str]:
    """反作弊七条的机器可检部分（1、2、3、5、7）。"""
    problems = []
    if report.accuracy.get("dE_meV_atom", 0.0) > 0.1:
        problems.append("第1条：能量漂移 > 0.1 meV/atom，加速比按 0 计")
    if report.accuracy.get("dF_meV_A", 0.0) > 1.0:
        problems.append("第1条：力漂移 > 1 meV/Å")
    for key in ("pseudo_sha256", "xc_libxc", "ecut", "kmesh"):
        if key not in report.inputs:
            problems.append(f"第2条：输入缺少 {key}")
    if report.wall_clock_s <= 0:
        problems.append("第3条：未记录端到端墙钟")
    if report.ledger:
        problems.append("第7条：存在降精度开关，必须在同一份报告里声明漂移")
    return problems


def run_case(name: str, case_kwargs: dict, budget: dict) -> RunReport:
    rep = RunReport(
        case=name,
        inputs={
            "pseudo_sha256": "synthetic",
            "xc_libxc": ["XC_LDA_X", "XC_LDA_C_PZ"],
            "ecut": case_kwargs.get("ecut", 2.0),
            "kmesh": "gamma",
        },
    )
    AccuracyLedger.reset()
    t0 = time.perf_counter()
    rep.factors.append(FactorReport("1 scf", "iterations", measure_scf_iterations(case_kwargs), budget["iterations"], "steps"))
    rep.factors.append(FactorReport("2 eigensolve", "matvec", measure_matvec_count(case_kwargs), budget["matvec"], "calls"))
    rep.wall_clock_s = time.perf_counter() - t0
    rep.ledger = AccuracyLedger.report()
    return rep


if __name__ == "__main__":
    case = dict(lattice=jnp.eye(3) * 6.0,
                frac=jnp.array([[0.0, 0.0, 0.0], [0.53, 0.47, 0.49]]),
                ecut=2.0, nband=2, density_factor=4.0)
    rep = run_case("synthetic-2atom-gamma", case, {"iterations": 20.0, "matvec": 30.0})
    print(rep.to_json())
    issues = check_anticheat(rep)
    print("反作弊检查：", issues if issues else "通过")
