# E2b: Matrix-free Davidson (production candidate)

## 假设

用预处理 block Davidson + matrix-free `Hψ` 替代 dense eigh；大 `nG` 时
dense eigh 是唯一主瓶颈。

## 实现

- `src/addft/solver/davidson.py`
- 初始子空间：单位矩阵列（identity）
- 每步：`Hψ(Q)` → small Ritz → 预处理残差 → 全重正交 → QR 扩展
- `max_sub=24`（4×nband），cold 15 步 / warm 7 步；残差 >1e-7 时 cold 重试
- SCF：`nG>4000` 自动走 Davidson；2 原子用 Broyden，大 cell 用 Kerker
- 测试：`tests/test_davidson.py`

## 100 Ry 结果

| 项 | dense eigh | Davidson |
|---|---:|---:|
| SCF iterations | 10 | 10 |
| SCF wall（warm） | 35.4 s | **~1.2 s** |
| 首个 eigh/步骤 | 2.36 s | ~0.2–0.5 s |
| 精度 vs ABACUS | — | dE 0.0355 meV/atom，dF 0.0092 meV/Å，dσ 0.0004 kbar |

Davidson 从 identity 初值可收敛到 `res ~ 1e-12`；100 Ry 已替换 dense eigh。

> 1.35 s 是在修复 “每次 solve_scf 重新 jit Davidson” 的编译缓存 bug、
> 缓存 Hψ、只对新列做 Hψ、改用 ABACUS 预条件、max_sub=24/cold15/warm7
> 之后；最初版本是 6.9 s。

## 16 原子修复

原 `max_sub=48` 对 `nband=34` 只有 1.4×nband，Davidson 残差停在 ~0.06。
改为 `max_sub=max(48, 4*nband)=136` 后，16 原子 30 次 SCF 收敛，
auto 阈值降到 `nG>3000`，SCF 43.2s → 7.1s。

## 限制

- 16 原子仍比 ABACUS 8 MPI（0.76s）慢约 6.8×；
- 小体系继续 dense eigh（延迟更低、精度更稳）；
- 100 Ry 仍比 ABACUS 1 核慢约 10×；下一步 FP32/fused Hψ 和更好 mixer。

## 决策

**Green（大 nG 路径）**。这是当前 dense eigh 的实际替代；小 nG 保留 dense
fallback，不再作为百原子目标。


## 分项（100 Ry）

| 项 | 时间 |
|---|---:|
| `Hψ`（6 bands） | 1.59 ms |
| effective potential | 1.49 ms |
| density | 1.40 ms |
| Davidson cold 15 | 88 ms |
| Davidson warm 5 | ~30 ms |
| Broyden update | 6 ms |
| **SCF warm** | **2.09 s** |
| E+F | 2.33 s |
| stress | 2.24 s |

## 为什么仍比 ABACUS 慢

ABACUS 1 核只做 **34 次 hPsi**（10 次 SCF，约 3.4/迭代），
ad-dft 当前 10 次 SCF 仍执行约 **1 cold 15 + 9 warm 7 + 4 polish 7**
≈ 106 次 Davidson 迭代，且每次 Hψ 作用在增长到 48 列的子空间上；
等效 6-band matvec 约 500–600 次。

虽然 ad-dft 单次 6-band Hψ 是 1.59 ms（比 ABACUS hPsi 的 ~5.3 ms 快），
但算法上多用了约 30 倍 matvec，加上 QR/重正交/Broyden/Python 循环，
最终约 1.2 s，仍比 ABACUS 0.68 s 慢约 1.8×。
