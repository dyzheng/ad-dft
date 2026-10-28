# E1: Matrix-free Hψ

## 假设

不构造 dense H，直接做 `Hψ`，去掉 100 Ry 的 `O(nG³)` dense eigh 基础。

## 实现

- `src/addft/ops/hpsi.py`
- `Tψ + V(r)ψ + V_NLψ`
- `V(r) = V_loc + V_H + V_xc` 来自现有 `effective_potential`
- 非局域用缓存的 `proj/D`
- 测试：`tests/test_hpsi.py`

## 正确性

与 dense `H @ ψ` 对照：

| 案例 | nG | max diff |
|---|---:|---:|
| Si/Γ 20 Ry | 411 | 1.4e-14 |
| Si/Γ 100 Ry | 4477 | 7.1e-13 |

## 性能

| 案例 | matrix-free `Hψ` | dense `Hψ` |
|---|---:|---:|
| 20 Ry | **0.85 ms** | 7.9 ms |
| 100 Ry | **1.26 ms** | 706.9 ms |

100 Ry 的 dense `Hψ` 就是 dense H 的 GEMM 成本，比 dense eigh 2.36s 小，
但仍远高于 matrix-free。

## Gate

- FP64 `Hψ` <2 ms @100 Ry：**通过（1.26 ms）**
- FP32/TF32 `Hψ` <1 ms：**未做**，下一轮 E4

## 决策

**Green（FP64 基础版）**。进入 E2 Davidson/LOBPCG；同时把 FP32 融合列为 E4。
