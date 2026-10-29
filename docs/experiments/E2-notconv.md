# E2d: notconv (only unconverged bands)

## 假设

按 ABACUS `Diago_DavSubspace` 的 `notconv` 策略，只对未收敛 band 扩展子空间，
减少 Hψ 列数。

## 原型

- Python 动态 shape：每轮按 `||r_i|| > tol` 选 band；
- 只对 `R[:, unconv]` 做预条件/QR/Hψ；
- 其余逻辑与 Davidson 相同。

## 结果（100 Ry）

| 初值噪声 | Hψ 列数 | Davidson 迭代 | residual | 时间 |
|---:|---:|---:|---:|---:|
| 1e-3 | 136 | 26 | 3.6e-9 | 10.2 s |
| 1e-2 | 163 | 33 | 8.2e-9 | 5.2 s |

残差/本征值正确，但 Python 动态 shape 导致 JAX 反复编译/同步，单次求解
反而 5–10s，不能进入生产。

## 结论

- notconv 的数学方向正确；
- 需要 **jitted fixed-block + mask/lax.switch**，或 C++/CUDA 动态块实现；
- 当前先用固定 block（warm=5/cold=15）和缓存 Hψ；
- 下一步可做 jitted notconv 或直接最小化。

## 决策

**Yellow**：保留为下一轮 P1 优化项，不进入主路径。
