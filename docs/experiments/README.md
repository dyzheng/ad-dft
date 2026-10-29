# 10× 实验台账

> 阶段总结：`docs/PHASE_SUMMARY.md`

对应计划：`docs/DEV_PLAN_10X.md`

## 状态

| ID | 方案 | 状态 | 记录 |
|---|---|---|---|
| E0 | ABACUS 1 核/GPU 基线 | green | `E0-baseline.md` |
| E1 | Matrix-free Hψ | green | `E1-hpsi.md` |
| E2 | Davidson warm start | green* | `E2-davidson.md` |
| E2c | Self-evolution of Davidson/mixer | green | `E2-evolution.md` |
| E2d | notconv dynamic-band prototype | yellow | `E2-notconv.md` |
| E2a | LOBPCG/Chebyshev/Krylov | yellow | `E2-lobpcg.md` |
| E3 | Direct Mermin minimization | yellow | `E3-direct-min.md` |
| E4 | FP32/TF32 + tensor cores | yellow | `E4-fp32-eigh.md` |
| E5 | Compressed/ALB/PAO | parked | 需独立基组 track |
| E6 | Density-matrix minimization | yellow | `E6-density-matrix.md` |
| E7 | Divide-and-conquer / scaling | active | `E7-scaling.md` |
| E8 | Persistent/CUDA Graph | partial | device loop/native pack 已做 |
| E9 | PAW/NAO | parked | 契约变化 |
| E10 | TT/neural | parked | 研究 |

状态：`todo` / `in-progress` / `green` / `yellow` / `red` / `parked`

## 实验记录模板

```markdown
# EXP-XX: <name>

## 假设
一句话：去掉什么成本，为什么更快。

## 输入/环境
- 案例、ecut、nG/nfft、GPU、CUDA/JAX、ABACUS commit
- 命令

## 实现
- 文件/分支/feature flag
- 关键公式或 kernel 设计

## 正确性
- vs JAX 参考 / ABACUS / FD 力 / 应力 / I-15

## 性能
- warm、median、3 次
- 分项时间、显存、launch 数

## Gate
- Green / Yellow / Red，对照 `DEV_PLAN_10X.md`

## 决策
- 合入 / 继续 / 回退，原因

## 复现
```bash
...
```
```
