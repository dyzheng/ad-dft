# E2: Warm LOBPCG with matrix-free Hψ

## 假设

用上一步波函数 + 少量 LOBPCG 步，在 <20 matvec 内代替 dense eigh。

## 实现

原型 `/tmp/test_lobpcg_mf.py`：

- 子空间 `[X, R, P, R_random]`，QR + Rayleigh–Ritz；
- 动力学对角预条件 `1/(λ - T_diag)`；
- 矩阵自由 `Hψ`；
- 固定步数，残差超阈值则 fallback dense eigh（尚未接入主路径）。

## 结果

### 20 Ry (`nG=411`)，初值 = exact + 1e-3 噪声

| LOBPCG 步数 | eig err | residual |
|---:|---:|---:|
| 1 | 6.6e-5 | 1.3e-2 |
| 3 | 7.1e-7 | 1.0e-3 |
| 5 | 8.9e-8 | 3.3e-4 |

5 步约 35 ms（含多次 `Hψ` + QR）。

### 100 Ry (`nG=4477`)，初值 = exact + 1e-3 噪声

| LOBPCG 步数 | eig err | residual |
|---:|---:|---:|
| 1 | 9.8e-2 | 1.13 |
| 3 | 9.4e-4 | 7.8e-2 |
| 5 | 3.2e-5 | 8.0e-3 |

5 步约 63–67 ms。初值误差 1e-2 时残差更差（0.3–1.0）。

## 结论

- 矩阵自由 `Hψ` 本身很快（1.26 ms @100 Ry）；
- 但 LOBPCG 在 rank/条件数上仍不够稳：
  - 需要更多步才能到 1e-8；
  - 初值误差 1e-2 时收敛很差；
  - `[X,R,P,random]` 的随机补秩不能根本解决简并/秩亏。
- 不能直接替换 dense eigh。

## 决策

**Yellow**：保留为 warm-start 候选，不进入主路径。下一步：

1. 更稳健的 Davidson（rank-revealing QR/SVD + soft locking）；
2. 或 E3 直接最小化 Mermin 自由能（不求解本征问题）；
3. 100 Ry 仍以 dense eigh 兜底。

## 追加：Chebyshev 滤波

- 20 Ry：degree 20 残差 7.4e-4；degree 30 残差 4.4e-4。
- 100 Ry：degree 20 残差 9.3e-3；degree 30 残差 5.2e-3。
- 时间：100 Ry degree 20 约 60–72 ms。
- 结论：仍然不够，且高阶多项式成本上升。

## 追加：block Krylov + Rayleigh–Ritz

- 固定块数 `m`，`S=[X,HX,...,H^mX]`，QR + Ritz。
- 100 Ry：
  - m=8 残差 1.6e-2；
  - m=12 残差 6.3e-3；
  - m=16 残差 2.9e-3；
  - m=12 时间约 70 ms。
- 20 Ry m=16 残差 2.4e-4。
- 结论：没有预条件时收敛仍慢；需要 shift-invert / Lanczos / 更强预条件。

## 最终决策

**Yellow**：所有低成本 warm eigensolver 变体都未达到生产精度；
主路径继续 dense eigh，下一步要么：
1. 实现 Lanczos/shift-invert + 预条件；
2. 或 E3 的 Riemannian L-BFGS/trust-region；
3. 或 E5 压缩基组降低 `nG`。
