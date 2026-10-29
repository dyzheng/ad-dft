# E7: Si supercell scaling (Phase 0)

## 目标

量化当前平面波 + dense eigh 架构在 2/8/16 原子上的标度，并确认 100 原子
必须换算法/表示。

## 输入

- Si 超胞，`ecutwfc=20 Ry`，Γ 点，`nbands=2N+2`；
- ABACUS 1 MPI / 1 OpenMP；
- ad-dft GPU，device/auto SCF；
- SCF-only（避免 16 原子在 8GB 上 E+F OOM）。

## 结果（20 Ry）

| atoms | nG | nfft | iters | ABACUS 1-core | ad-dft SCF | 比值 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 411 | 13824 | 9 | 0.215 s | 0.436 s | 2.0× |
| 8 | 1591 | 58320 | 30 | 0.616 s | 8.52 s | 13.8× |
| 16 | 3143 | 91125 | 30 | 1.67 s | 35.4 s | 21.2× |

## 发现

1. **device Broyden 在 8 原子失效**：300 步后残差 ~0.15；Python DIIS
   可收敛但 40+ 步。现已对 `natom>2, nG>800` 自动切到 Python Kerker
   (`alpha=0.7, gg0=0.5`)，8/16 原子都能 30 步收敛。
2. **dense eigh 标度不可用**：`nG ∝ N`，eigh `O(N³)`；16 原子已 35s，
   100 原子 20 Ry 估算 ~数十分钟/次 eigh，100 Ry 的 dense H 约 800GB。
3. **8GB 显存是硬限制**：16 原子 SCF 可以，但 envelope E+F 已 OOM；
   100 原子必须 sparse/localized/多卡。
4. ABACUS 1 核在 16 原子仍 1.67s；当前 ad-dft 差距随 N 增大，不是收敛。

## 决策

- **E7 从 parked 转为 active**：百原子目标必须走局域化/分治/线性 scaling。
- 短期：保持 2 原子 device Broyden；大 cell 用 Kerker；继续找更好的 mixer。
- 中期：fragment/局域轨道/压缩基，目标是近线性；不能继续 dense eigh。

## 追加：16 原子 vs ABACUS MPI 8 进程（20 Ry）

| 平台 | SCF wall |
|---|---:|
| ABACUS CPU 1 核 | 1.86 s |
| ABACUS MPI 4 | 0.79 s |
| ABACUS MPI 8 | **0.76 s** |
| ad-dft GPU `auto`（修复前，dense） | 43.2 s |
| ad-dft GPU `auto`（修复后，Davidson） | **5.2 s**（32 iters，res 9.3e-11，E 正确） |

结论：

- 16 原子、`nG=3143`、`nband=34` 时，原 `max_sub=48` 只有 1.4×nband，
  Davidson 无法收敛；
- 改为 `max_sub=max(48, 4*nband)=136` 后，16 原子 32 次迭代收敛，
  `auto` 阈值降到 `nG>3000`，SCF 从 43.2s 降到 **5.2s**；
- `eig_tol` 从 1e-7 放宽到 1e-3，减少不必要的 cold 重试；
- 但仍比 ABACUS 8 MPI 的 0.76s 慢约 **6.8×**，比 1 核 1.86s 慢约 2.8×；
- E 与 ABACUS 仍差约 4.5 meV/atom，需单独复核；
- 必须先把 Davidson/mixer 修到 16 原子可用，或走局域化/线性 scaling；
- ad-dft auto 的能量与 ABACUS 差约 **4.5 meV/atom**，也要单独复核。
