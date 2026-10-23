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
| L2 原语（密度/Hartree/XC/V_NL/Ewald/占据） | ✅ UPF v1/v2、真实 m 通道、libxc LDA；NLCC 显式拒绝 |
| L3 求解器（DIIS-SCF、稠密 eigh、抛光到不动点） | ✅ Γ + 联合壳层非 Γ；dense eigh 仍为 V0 限制 |
| L4 微分层（变分包络 + 不动点伴随） | ✅ 能量导数冻结 ψ、ρ(geo,ψ) 随几何重建；matrix-free 伴随用于非能量观测量 |
| L5 观测量（能量分解 / 力 / 应力 / ΔQ） | ✅ 力/应力/能量与固定二进制 ABACUS 对照通过 |

**关键数字**：Si/Γ 总能 vs 固定二进制 ABACUS 差 `0.277 meV/atom`；
Si 畸变构型力最大差 `0.593 meV/Å`；Si/Γ 应力最大差 `0.029 kbar`。
I-15 在 `nfft=3375` 上 `corr=0.99999999999946`、`rel=1.14e-6`；
`ΔQ` 伴随残差 `2.28e-11`；伴随成本 `2.22×` SCF、`nfft=3375` RSS 为
`nfft=729` 的 `0.88×`；力 vs FD 在 h=1e-3 时 `5.66e-7 Ha/Bohr`（∝h²）。
详见 `docs/reports/M0.md`。

```bash
export PYTHONPATH=src:tests
python -m pytest tests                        # 52 passed, 1 skipped, 0 xfail
python -m pytest tests/test_forces.py -m slow # 5 passed
```

`tests/test_upf.py::test_against_abacus_element_info` 在未设置
`ADDFT_RUN_ABACUS=1` 时 skip。固定二进制 ABACUS v3.11.0-beta.1
（commit `c17cbac9b`）已在本机用同一 INPUT 重跑并写入
`tests/regress/*/OUT.*`；需要现场复核时：

```bash
ADDFT_RUN_ABACUS=1 python -m pytest tests/test_abacus_regression.py -m abacus
```

## 快速开始

```python
import addft as ad
import jax

s = ad.setup(lattice=[[6.,0,0],[0,6.,0],[0,0,6.]],
             frac=[[0,0,0],[0.53,0.47,0.49]],
             ecut=2.0, nband=2, density_factor=4.0)

E, F = ad.energy_and_forces(s)        # 力 = dE/dR 经隐式伴随
sigma = ad.stress(s)                  # σ = −(1/Ω) dF/dε（ABACUS 约定）
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
| `docs/reports/M0.md` | R3/M0 出口报告、误差预算、GO/NO-GO |
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
