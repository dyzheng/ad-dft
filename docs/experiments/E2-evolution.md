# E2c: Self-evolution loop for Davidson/mixer controls

## 目标

把 Davidson/Broyden 的控制参数变成可搜索变量，闭环评估并进化，而不是手工调参。

## 可搜索维度

- `ADDFT_DAVID_MAX_SUB`：24/36/48/60
- `ADDFT_DAVID_COLD`：10/15/20/25
- `ADDFT_DAVID_WARM`：5/7/10/12
- `ADDFT_DAVID_PRECOND`：`abacus` / `kinetic`
- `ADDFT_DAVID_PRECOND_ALPHA`：0.5/0.75/1.0/1.25/1.5（预条件指数）
- `ADDFT_BROYDEN_HI`：0.7/0.8/0.9
- `ADDFT_BROYDEN_LO`：0.4/0.5/0.6

## 评估

- 100 Ry Si/Γ，`nG=4477`
- 每个候选在独立进程，先 warmup 编译，再计时一次 SCF
- 目标：SCF wall + residual/accuracy penalty
- 结果写入 `bench/results/evolution.json/md`

## 结果

| max_sub | cold | warm | precond | Broyden | iters | SCF |
|---:|---:|---:|---|---:|---:|---:|
| 48 | 15 | 7 | abacus | 0.9→0.6 | 10 | 1.312 s |
| 24 | 15 | 7 | abacus | 0.9→0.6 | 10 | **1.310 s** |
| 36 | 15 | 7 | abacus | 0.9→0.6 | 10 | 1.323 s |
| 48 | 15 | 7 | kinetic | 0.9→0.6 | 15 | 1.945 s |
| 48 | 25 | 7 | kinetic | 0.9→0.6 | 56 | 4.909 s |

### 预条件指数 α

演化测试 α=0.5/0.75/1.0/1.25/1.5：

- α=1.0（ABACUS 公式）最好；
- α=0.5：27 次迭代、4.12s；
- α=1.5：14 次迭代、1.61s；
- 没有找到优于 α=1.0 的组合。

结论：

- ABACUS 形式预条件明显优于 `1/(λ-T)`；
- `max_sub=24`（4×nband）已足够，和 48 基本持平；
- Broyden 0.9→0.6 是 2 原子 100 Ry 的稳定组合；
- kinetic 预条件/大 cold 步数会把迭代数推高。

## 运行

```bash
python bench/evolve.py --generations 3 --population 6 --elite 2 --seed 3
```


## 公平基线补充

| 平台 | 100 Ry SCF/整体 |
|---|---:|
| ABACUS CPU 1 核 | 0.67 s（含力/应力） |
| ABACUS MPI 4 | 0.35 s |
| ABACUS MPI 8 | 0.42 s |
| ABACUS MPI 20 | 2.50 s（该构建线程配置不佳） |
| ABACUS GPU（v3.10.1） | 2.51 s（含力/应力） |
| ad-dft GPU | 1.23 s SCF / 2.33 s E+F |
| ad-dft CPU 1 线程 | 10.97 s SCF |

因此当前“GPU vs ABACUS 1 CPU 核”仍不能证明 10×；对 4 核 ABACUS，
ad-dft E+F 约慢 6.7×。
