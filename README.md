# ad-dft · 自动微分周期性 DFT 内核（架构原型）

一个**可微的周期性平面波 DFT 内核**的架构骨架：能量是唯一可微标量，
力/应力/极化/约束乘子都是它的导数；SCF 用不动点隐式微分，显存与迭代数解耦。

复刻对象是 **ABACUS 的物理契约**（不是它的代码）。所有约定写在 `spec/`，
所有判据写在 `docs/VALIDATION.md`，所有未完成项写在 `docs/TASKS.md` 与
`ARCHITECTURE.md §8`。

## 状态（诚实版）

| 层 | 状态 |
|---|---|
| L0 运行时（单位/dtype/精度账本） | ✅ |
| L1 几何（Cell 可微、冻结壳层、FFT 盒规则） | ✅ 盒规则与 ABACUS 逐位一致 |
| L2 原语（密度/Hartree/XC/V_NL/Ewald/占据） | ✅ 除 V_NL 的 Hermitian 诊断项 |
| L3 求解器（DIIS-SCF、稠密 eigh、抛光到不动点） | ✅ 73 步收敛到 2e-13 |
| L4 微分层（包络定理 + 不动点伴随） | ✅ 响应算子 corr 1.0000 |
| L5 观测量（能量分解 / 力 / 应力 / ΔQ） | ✅ 力 vs Richardson 外推 3.1e-12 |

**关键数字**：双能量形式差 0（机器精度）；`ΣF = 4.2e-8`；力 vs FD 在 h=1e-3 时
5.7e-7 Ha/Bohr（∝h²），Richardson 外推后 3.1e-12。

已修 5 个"能量对、梯度错"的 bug（详见 `docs/ARCHITECTURE.md §8`），
剩 3 项诊断缺口（不阻塞物理，但必须在 K04/K08 清掉）。

```bash
export PYTHONPATH=src:tests
python -m pytest tests/test_contracts.py    # 6 passed
python -m pytest tests/test_primitives.py   # 7 passed, 2 xfailed（诊断项）
python -m pytest tests/test_invariants.py   # 6 passed
python -m pytest tests/test_forces.py       # 2 passed（slow 门禁见 VALIDATION §8.3）
```

`xfail(strict=True)` 是刻意的：一旦某个缺口被修好，测试会从 xfail 变成
"unexpectedly passing" 从而**让 CI 失败**，迫使你更新记录。

## 快速开始

```python
import addft as ad
import jax

s = ad.setup(lattice=[[6.,0,0],[0,6.,0],[0,0,6.]],
             frac=[[0,0,0],[0.53,0.47,0.49]],
             ecut=2.0, nband=2, density_factor=4.0)

E, F = ad.energy_and_forces(s)        # 力 = dE/dR 经隐式伴随
sigma = ad.stress(s)                  # σ = (1/Ω) dE/dε
terms = ad.energy_decomposition(s)    # 带 tag 的七项分解，可逐项开关
```

## 文档地图

| 文件 | 内容 |
|---|---|
| `spec/units.md` | 单位制、Ry↔Ha 桥、泛函导数的 (Ω/N) 权重 |
| `spec/discretization.md` | 壳层冻结、FFT 盒规则、**表示一致性** |
| `spec/pseudopotential.md` | UPF 索引约定、为什么不预表化、NLCC 决策 |
| `spec/occupation.md` | 占据数、μ 的标量隐式、F=E−TS |
| `spec/invariants.md` | 15 条不变量 + 噪声地板 + 差异预算 |
| `docs/ARCHITECTURE.md` | 分层、可微边界、三条求导路径、已知缺口 |
| `docs/TASKS.md` | 22 个模块 + 5 个里程碑 + 任务卡模板 + 24 月节奏 |
| `docs/VALIDATION.md` | 五级阶梯、五步协议、反作弊七条、封存集、CI 门禁 |
| `docs/ABACUS_FACTS.md` | 从 ABACUS 源码逐条确认的事实（含文件行号） |

## 设计原则（七条，每条对应一个失败模式）

1. AD-native 但不是 AD-only：每个原语三份实现（ref / JAX / kernel）。
2. periodic-first：`Cell` 是可微叶子，`KMesh` 权重和为 1，投影随几何重建。
3. 收敛与微分解耦：L4 只依赖收敛态，不依赖迭代历史。
4. 观测量是能量的导数，不是独立模块。
5. 性能关键路径下沉，物理组合上浮。
6. 可验证性优先于性能：梯度门禁与性能门禁并列。
7. 契约先行：涉及离散化/泛函/赝势/单位/边界条件的改动，先改 `spec/`。
