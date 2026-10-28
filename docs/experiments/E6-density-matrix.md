# E6: Density-matrix purification (McWeeny)

## 假设

不做 eigh，用 `P²=P` 的密度矩阵纯化 + GEMM 代替本征分解。

## 原型

- 20 Ry，`nG=411`，4 条占据带。
- 初始 `P0 = 0.5 I + (μI − H)/(2Δ)`，`Δ=(λmax−λmin)/2`。
- McWeeny：`P ← 3P² − 2P³`。
- μ 由能隙中心给出（原型中用 exact eigh 的 HOMO/LUMO；生产必须用无 eigh 的 μ 搜索）。

## 结果（20 Ry）

| 迭代 | idempotency | trace | Pdiff vs eigh |
|---:|---:|---:|---:|
| 5 | 0.245 | 16.1 | 0.308 |
| 10 | 0.189 | 4.82 | 0.271 |
| 15 | 0.036 | 4.0004 | 0.038 |
| 20 | **1e-15** | **4.0000** | **2.7e-14** |

时间：20 次迭代约 0.08–0.16 s（`nG=411`），比该规模 dense eigh（22 ms）慢。

## 100 Ry 成本

`nG=4477` complex GEMM：

| dtype | 单次 `P@P` |
|---|---:|
| complex128 | **1.80 s** |
| complex64 | **0.033 s** |

McWeeny 每次迭代需要 2 次 GEMM：

- FP64：~3.6 s/迭代，比 dense eigh 2.36 s 更慢；
- FP32：~0.066 s/迭代，20 次 ~1.3 s，只有约 1.8× 提升，且精度/μ 搜索仍是问题；
- 显存：多个 `nG²` 矩阵，100 Ry 下接近 GB 级。

## 结论

- 纯化本身正确，可作为无 eigh 的密度矩阵方法；
- 稠密纯化在小体系和 FP64 下不划算；
- 只有在 **localized/sparse P + 线性 scaling + FP32 tensor cores** 时才有大体系价值；
- 当前 2 原子 Si 原胞不适用。

## 决策

**Yellow / 大体系 track**。短期不进入主路径；E7 线性 scaling 时再启用。
