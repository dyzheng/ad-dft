# 10× 性能开发计划（Dev Plan v1）

> 模块级阶段总结：`docs/PHASE_SUMMARY.md`

> 状态：执行中  
> 目标：在**同一赝势、同一 `ecut`、同一精度门禁**下，把 ad-dft 的
> `SCF + 力 + 应力` 做到比 ABACUS 快 10× 以上。  
> 关联文档：`docs/reports/ABACUS_COMPARISON.md`、`docs/reports/M1.md`、
> `native/README.md`

---

## 0. 目标定义（必须先锁定）

“10×” 必须写清楚基准，否则不可验证。

| 维度 | 定义 |
|---|---|
| 基线 A | ABACUS 1 MPI / 1 OpenMP 线程，同 PP、同 `ecutwfc`、同 k 点、同 `nband` |
| 基线 B | ABACUS GPU 构建，同输入 |
| 计时范围 | `SCF + cal_force + cal_stress` 的总 wall time |
| 精度 | 与 ABACUS 的 E/F/σ 差在现有门禁内；与 full-PW 参考一致 |
| 环境 | 固定 GPU 型号、驱动、CUDA、JAX 版本；记录 ABACUS commit |
| 样本 | 至少 20 Ry、40 Ry、100 Ry 三档；后续加大体系 |

当前 100 Ry 基准：

| 量 | ABACUS 1 核 | ad-dft GPU | 10× 目标 |
|---|---:|---:|---:|
| SCF wall | 0.68 s | 35.2 s | **< 0.068 s** |
| 迭代数 | 10 | 10 | — |
| dense eigh | — | 2.36 s × 15 = 35.4 s | 去掉 |

**判定：**
- **Green**：同输入 E/F/σ 达标，wall time ≤ 基线/10。
- **Yellow**：wall time ≤ 基线/3，且各分项有明确优化路径。
- **Red**：关键分项（如单次 `Hψ`、单次直接最小化评估）达不到预算，停止该方案。

> 如果目标是“比 ABACUS GPU 快 10×”，难度显著高于 1 核基线；必须先完成
> 基线 B 的实测，不能拿 1 核数字替代。

---

## 1. 当前基线（100 Ry）

`nG=4477`、`nG_rho=35749`、`nfft=110592`、`nband=6`。

| 步骤 | 单次 | 调用数 | 总计 | 占 SCF |
|---|---:|---:|---:|---:|
| **dense eigh** | **2.36 s** | 15 | **35.4 s** | **~99%** |
| H 组装 | 9 ms | 15 | 0.135 s | 0.4% |
| Hartree | 2 ms | 15 | 0.03 s | 0.1% |
| density | 2 ms | 15 | 0.03 s | 0.1% |
| XC | 2 ms | 15 | 0.03 s | 0.1% |
| geometry cache | 0.13 s | 1 | 0.13 s | 0.4% |

15 次 eigh = 10 次主循环 + 4 次 polish + 1 次最终本征求解。

**结论：**
1. dense eigh 是唯一主矛盾；
2. 去掉 eigh 后，单步剩余成本只有 ~15–20 ms；
3. 真正的 10× 需要把算法、表示、精度、融合四层同时推进。

---

## 2. 非妥协约束

任何方案必须满足：

1. 同一赝势、同一 `ecutwfc`、同一 k 点、同一 `nband`、同一占据；
2. E/F/σ 与 ABACUS 的差值不劣于现有门禁；
3. I-15 响应、FD 力/应力、ΔQ 伴随门禁不退化；
4. JAX 参考路径始终保留，native 后端可关闭；
5. 所有实验有独立开关、可回退、有 benchmark 日志；
6. 不允许“静默换基组/换赝势”来刷性能；换基组必须单独立项。

---

## 3. 10× 预算（100 Ry）

目标 `< 0.068 s`（vs ABACUS 1 核 0.68 s）。

| 项 | 预算 | 备注 |
|---|---:|---|
| 本征求解/直接最小化（10–20 次 `Hψ`） | 20–40 ms | 单次 `Hψ` <1 ms |
| 密度 + XC + Hartree | 5–10 ms | 融合/低精度 |
| 力 + 应力 | 5–10 ms | envelope + 缓存 P/D |
| 混合/残差/调度 | <5 ms | device-side loop |
| **合计** | **35–65 ms** | 必须打进 |

**硬指标：**
- 100 Ry 单次 `Hψ` **<1 ms**（FP32/TF32，融合后）；
- 总迭代/评估次数 **<20**；
- 不能出现 dense `nG×nG` 矩阵或 dense eigh。

---

## 4. 方案优先级（按尝试顺序）

### P0. 基准、实验台、目标锁定（1 周）

**要做：**
- 建立 `bench/perf_10x.py`：同输入、warm、median、分项计时、精度对照；
- 测 ABACUS 1 核与 ABACUS GPU；
- Nsight Systems 抓 launch 数、kernel 时间、显存；
- 输出 `docs/experiments/E0-baseline.md`。

**Gate：**
- 20/40/100 Ry 数字可复现；
- 明确 10× 是相对 1 核还是 GPU。

---

### P1. Matrix-free Hψ + 融合算子（2–3 周，最高优先级）

**假设：** 不构造 dense H，只做 `Hψ`，是去掉 99% eigh 成本的前提。

**实现：**
```text
Hψ = FFT⁻¹[T_G ψ_G]
   + (V_loc + V_H + V_xc) ψ
   + Σ |β_i> D_ij <β_j|ψ>
```

- `src/addft/ops/hpsi.py`：JAX 参考实现；
- `native/hpsi.cu`：融合 kernel；
- cuFFTDx / 现有 native pack；
- 非局域投影用 batched complex GEMM，D 缩放融入 epilogue；
- FP64 先正确，FP32/TF32 后加速。

**Gate：**
- `Hψ` vs dense H 矩阵乘：相对误差 <1e-10（FP64）；
- 100 Ry 单次 `Hψ`：FP64 <2 ms，FP32/TF32 **<1 ms**；
- 显存不再包含 320 MB dense H。

> 当前 FP64 1.26 ms 已过 FP64 门禁；FP32/TF32 尚未实现，因此 E1 记为
> `green*`（FP64 绿，FP32 待 E4）。

**预期：** 为 P2/P3 提供基础；这是 10× 的必要条件。

**Kill：** 若 FP32 `Hψ` 仍 >3 ms，先优化 kernel，不进入算法阶段。

---

### P2. Davidson/LOBPCG + warm start + 自适应子空间（2–4 周）

**假设：** 用上一步波函数 + 少量 matvec，可在 <20 次内得到 lowest nband。

**实现：**
- `src/addft/solver/davidson.py`；
- 初始子空间：上一步 ψ + 预处理残差；
- rank-revealing QR/SVD、soft locking、简并处理；
- 固定 `max_subspace`，device-side；
- fallback dense eigh（小体系/不收敛）。

**Gate：**
- 100 Ry 残差 <1e-8，matvec <20；
- 与 dense eigh 的本征值/子空间一致；
- SCF 不含 dense eigh。

**预期：** 去掉 dense eigh；100 Ry SCF 从 35 s 降到 0.2–1 s 量级。

**Kill：** >40 matvec 仍不收敛 → 转 P3/P5。

---

### P3. 直接最小化 Mermin 自由能（3–6 周，战略重点）

**假设：** 不做 SCF、不做 eigh，直接优化 `C, f`，可减少外循环和 charge sloshing。

**目标函数：**
```text
F[C,f] = E[C,f] − σ S[f]
C^H S C = I
Σ f = N
```

**实现：**
- `src/addft/solver/direct_min.py`；
- Riemannian L-BFGS / trust-region；
- 预条件 `(T + α)⁻¹`；
- retraction：QR/极分解；
- 占据：Fermi–Dirac + Newton(μ)；
- envelope theorem 算力/应力。

**Gate：**
- 能量/梯度评估 <15 次；
- E/F/σ 与 ABACUS 一致；
- 金属/小 gap 有 fallback；
- 100 Ry SCF-free wall time 进入 0.1–0.3 s，冲 <0.068 s。

**预期：** 10× 的主要候选。若成功，P2 可作为兜底。

---

### P4. 混合精度 + tensor cores + 融合（与 P2/P3 并行）

**假设：** 3070 Ti 的 FP32 约为 FP64 的 64×；内层低精度可大幅提速。

**实现：**
- Hψ、非局域 GEMM、子空间 Ritz 用 FP32/TF32；
- 能量、残差、力/应力用 FP64 累加；
- 迭代精化（iterative refinement）；
- tensor cores 处理 complex GEMM（实部/虚部分拆）；
- 记录 `AccuracyLedger`。

**Gate：**
- E/F/σ 漂移 <门禁；
- 内层 2–5× 加速；
- 不引入 NaN/不收敛。

---

### P5. 压缩/自适应基组（4–8 周，高风险高回报）

**假设：** 波函数自由度可压缩 5–10×，同时保持变分性和力/应力正确。

**候选：**
1. PAO/NAO/localized orbital：`ψ = Φ c`，小广义本征问题；
2. Adaptive Local Basis (ALB/DGDFT)；
3. Selected G-space / 自适应壳层；
4. 低秩/Tensor-train（研究）。

**Gate：**
- 自由度降低 ≥5×；
- E 差 <1 meV/atom，F 差 <10 meV/Å；
- 对 Si/畸变/Si 40/100 Ry 都可复现；
- 力/应力仍由 envelope theorem 得到。

**风险：** 基组不完备、迁移性、契约变化。必须单独 track，不能静默替换。

---

### P6. 密度矩阵最小化/纯化（4–8 周，备选主线）

**假设：** 优化 `P` 而不是 `ψ`，可绕过 eigh 和 SCF，并自然线性 scaling。

**实现：**
- `P = f(H)`：McWeeny 纯化 / sign function；
- 有限温：Fermi–Dirac 熵；
- 大体系：localized/sparse P；
- GPU batched GEMM。

**Gate：**
- 无 eigh；E/F/σ 达标；
- 大 N 近线性；
- 与 P3 对比，选更快者。

---

### P7. 分而治之 / 线性 scaling（6–12 周，大体系）

**假设：** 大体系用 fragment + embedding + batched small eigensolve，可近线性。

**实现：**
- 实空间分块/fragment；
- 片段间 embedding/poison；
- GPU batched small eigensolve；
- 与 P6 的 localized P 结合。

**Gate：**
- N 增大 10×，wall time <3×；
- 精度不退化；
- 小 Si 原胞不强制使用。

**注意：** 对当前 2 原子 Si 无用；这是大体系 10× 的来源。

---

### P8. GPU 调度/融合/多流（持续）

**内容：**
- persistent kernel、CUDA Graph；
- 多 stream：FFT/GEMM/band/k-point；
- 多 GPU；
- 内存布局：SoA、band-major、G-shell 排序；
- 减少 host sync。

**Gate：**
- launch/调度开销 <5%；
- kernel 融合后 global memory 往返显著下降。

---

### P9. 换基组/PAW/NAO（条件触发）

**只在 P1–P5 仍达不到 10× 且允许改变契约时启动。**

- PAW/USPP 可在同精度下降低 `ecut`；
- NAO 可大幅降低 nG；
- 但这**不能**与“同 PP 对照 ABACUS”混在一起；
- 需要独立精度基准和力/应力验证。

---

### P10. 研究型方案（低优先级）

- Tensor-train / CP 分解波函数；
- 神经/学习基；
- 量子蒙特卡洛式表示。

只在 P1–P7 全部失败时投入；不作为生产依赖。

---

## 5. 实验台账（Living Tracker）

| ID | 方案 | 状态 | 原型/脚本 | 关键指标 | 结果 | 决策 |
|---|---|---|---|---|---|---|
| E0 | ABACUS 1 核/GPU 基线 | green | `docs/experiments/E0-baseline.md` | wall, E/F/σ | 20/40/100 Ry 已记录 | 继续 E1 |
| E1 | Matrix-free Hψ | green* | `docs/experiments/E1-hpsi.md` | FP64 <2 ms / FP32 <1 ms | FP64 1.26 ms；FP32 待做 | 进入 E2/E4 |
| E2 | Davidson warm | yellow | `docs/experiments/E2-lobpcg.md` | <20 matvec | 5 步残差 8e-3 @100 Ry | 转 E3/E5 |
| E3 | Direct Mermin min | yellow | `docs/experiments/E3-direct-min.md` | <15 evals | 简单梯度 3 步后停滞 | 需 L-BFGS/trust-region |
| E4 | FP32/TF32 + tensor cores | yellow | `docs/experiments/E4-fp32-eigh.md` | 2–5× | FP32 eigh 3.5×，精度不足 | 需 Chebyshev/Lanczos 精化 |
| E5 | Compressed/ALB/PAO | parked | 需独立基组 track | 5× DOF, <1 meV | — | 契约变化 |
| E6 | Density-matrix min | yellow | `docs/experiments/E6-density-matrix.md` | no eigh, linear | 20 Ry 可达 1e-14；100 Ry FP64 GEMM 1.8s | 仅大体系/稀疏 P |
| E7 | Divide-and-conquer | parked | `solver/dc.py` | near-linear | — | 大体系再启用 |
| E8 | Persistent/CUDA Graph | partial | runtime | <5% overhead | device loop/native pack 已做 | 继续 CUDA Graph |
| E9 | PAW/NAO | parked | separate | 5–20× | — | 契约变化 |
| E10 | TT/neural | parked | research | 10×+ | — | 研究 |

**状态取值：** `todo` / `in-progress` / `green` / `yellow` / `red` / `parked`。

---

## 6. 实验协议（每个方案都必须走）

1. **假设**：一句话写清“去掉什么成本、为什么更快”。
2. **原型**：独立脚本/分支，不改主路径；固定输入。
3. **正确性**：与 JAX 参考、ABACUS、FD 力/应力对照。
4. **性能**：warm、median、至少 3 次；记录 GPU 型号/频率/显存。
5. **Gate**：按该方案的硬指标判定 Green/Yellow/Red。
6. **决策**：
   - Green → 合入 feature flag，补测试；
   - Yellow → 记录瓶颈，决定是否继续；
   - Red → 回退，写清楚失败原因和数据。
7. **文档**：`docs/experiments/EXP-<id>.md`，包含命令、日志、结论。

### 实验模板

```markdown
# EXP-XX: <name>

## 假设
## 输入/环境
## 实现
## 正确性
## 性能（warm/median）
## Gate 结果
## 决策（Green/Yellow/Red）
## 复现命令
```

---

## 7. 决策树

```text
E1 Hψ 达不到 <1ms
  └─ 优化 kernel/cuFFTDx/FP32，不进入 E2/E3

E2 Davidson 达不到 <20 matvec
  └─ E3 Direct minimization

E3 达不到 <15 evals
  └─ E6 Density matrix / E5 Compressed basis

E5 精度不达标
  └─ E9 PAW/NAO（若允许）

E3 + E5 都达标
  └─ E4/E8 融合，冲 10×

大体系扩展
  └─ E7 Divide-and-conquer + E6 linear scaling
```

---

## 8. 里程碑与节奏

| 阶段 | 周期 | 交付 |
|---|---:|---|
| P0 基线 | 1 周 | benchmark 脚本、E0 报告 |
| P1 Hψ | 2–3 周 | matrix-free Hψ + tests，单次 <1ms |
| P2 Davidson | 2–4 周 | warm Davidson/LOBPCG，<20 matvec |
| P3 Direct min | 3–6 周 | SCF-free Mermin 最小化 |
| P4 混合精度 | 2–4 周（并行） | FP32/TF32 + tensor cores |
| P5 压缩基 | 4–8 周 | PAO/ALB 原型 + 精度门禁 |
| P6 DM 最小化 | 4–8 周 | 无 eigh 的 P 优化 |
| P7 分治 | 6–12 周 | 大体系近线性 |
| P8 调度 | 持续 | persistent/CUDA Graph/多流 |

建议 3–4 人：1 名 GPU/kernel、1 名数值/算法、1 名 DFT/验证、
1 名 JAX/runtime。

---

## 9. 风险登记

| 风险 | 影响 | 对策 |
|---|---|---|
| 单次 Hψ 降不到 1ms | 10× 不成立 | cuFFTDx、融合、FP32、persistent |
| Davidson 不收敛 | 退回 SCF/eigh | warm start、rank-revealing、fallback |
| Direct min 条件数差 | 收敛慢 | 预条件、trust-region、SCF 兜底 |
| 压缩基精度不够 | 结果不可信 | 系统基组收敛测试、full-PW 对照 |
| 金属/小 gap | direct min 失效 | Mermin + 占据优化 + fallback |
| 换基组破坏契约 | 不可比 | 独立 track，不混入同 PP 对照 |
| GPU 显存不足 | dense H/P 太大 | matrix-free、分块、低精度 |
| ABACUS GPU 基线更强 | 10× 更难 | 先测基线，按 GPU 对 GPU 判定 |

---

## 10. 立即执行（未来两周）

1. 写 `bench/perf_10x.py`，固定 20/40/100 Ry 输入与计时协议。
2. 跑 ABACUS 1 核 + ABACUS GPU 基线。
3. 实现 JAX 参考版 matrix-free `Hψ`，与 dense H 对照。
4. 写 `native/hpsi.cu` 原型，测 100 Ry 单次耗时。
5. Nsight profile：launch 数、FFT/GEMM/非局域/点乘时间。
6. 若 `Hψ` <1ms，进入 Davidson；否则继续 kernel。

**当前不做的事：**
- 不再为 dense eigh 做微优化；
- 不把 Davidson 单独当作 10× 方案；
- 不在未验证精度前替换基组/赝势。

---

## 11. Cycle 1 结论（E0–E6）

| ID | 状态 | 关键结果 | 是否进入主路径 |
|---|---|---|---|
| E0 | green | 20/40/100 Ry 基线 | 是 |
| E1 | green* | matrix-free `Hψ` 1.26 ms @100 Ry（FP64） | 代码已落地，FP32 待做 |
| E2 | yellow | LOBPCG/Chebyshev/Krylov 最好残差 ~1e-3 | 否 |
| E3 | yellow | 简单梯度 0.1 mHa 后停滞 | 否 |
| E4 | yellow | FP32 eigh 3.5×，精度不足 | 否 |
| E5 | parked | 需独立基组 track | 否 |
| E6 | yellow | 20 Ry 可达 1e-14；100 Ry FP64 GEMM 不划算 | 否 |
| E7 | parked | 大体系才适用 | 否 |
| E8 | partial | device loop/native pack 已做 | 是 |
| E9/E10 | parked | 契约/研究 | 否 |

**当前事实：**
- 100 Ry 生产路径仍是 dense eigh，35.2 s；10× 未达成；
- E1 证明了 matrix-free 路线可行；
- 低成本 eigensolver 全部不够，下一步必须做二者之一：
  1. **Lanczos/shift-invert + 强预条件**（E2b）；
  2. **Riemannian L-BFGS/trust-region**（E3b）；
- E5 压缩基组是唯一可能直接改变 10× 量级的表示层方案，但需要契约决策；
- E7/E9/E10 不是当前小体系可验证的方案。

**停止条件更新：**
- 近中期可执行方案（E1–E6）已逐一尝试；
- E7/E9/E10 标记 parked，不阻塞当前 10× 主线；
- 下一 Cycle 从 E2b 或 E3b 继续。
