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

## 原生 GPU 后端（实验）

原生 CUDA 后端已包含 **LDA XC 融合 kernel、cuSOLVER eigensolver、
G-shell scatter/gather**；没有构建时自动回退到 JAX/libxc 路径：

```bash
NVCC=/root/miniconda/bin/nvcc ./native/build.sh   # CUDA >= 12
export PYTHONPATH=src:tests
python -m pytest tests/test_native_xc.py -q
```

`ADDFT_NATIVE_XC=0` 可显式关闭原生路径。当前实测（RTX 3070 Ti / WSL2）：
`bench/adjoint.py 3375` 的 SCF 从 `19.35 s` 降到 `0.34 s`，SCF+grad 从
`41.75 s` 降到 `3.19 s`；`nfft=729` 的 SCF 从 `32.94 s` 降到 `0.29 s`。
设备端 `lax.while_loop` SCF、原生 cuSOLVER eigensolver 与原生 XC 融合
kernel 均已接入；详见 `native/README.md` 与 `docs/reports/M1.md`。

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
| `docs/reports/M1.md` | 原生 GPU 后端、设备端 SCF、Broyden、native pack |
| `docs/reports/ABACUS_COMPARISON.md` | 20/40/100 Ry 与 ABACUS 的精度/性能/分项时间 |
| `docs/DEV_PLAN_10X.md` | 10× 性能路线、方案优先级、实验台账、决策门禁 |
| `docs/PHASE_SUMMARY.md` | 阶段总结：各模块设计考虑、尝试方案、实测效果、失败模式与下一步 |
| `docs/experiments/` | E0–E6 逐项实验记录（基线、Hψ、eigensolver、direct min、FP32、DM 纯化） |
| `bench/README.md` / `bench/dashboard.py` | 实时性能看板：对比 SCF/后端/驱动/原生开关 |
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
