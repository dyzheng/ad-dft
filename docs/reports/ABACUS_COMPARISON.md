# ABACUS 直接运行对照

> 目的：在**同一赝势、同一 INPUT 物理参数**下，比较 ad-dft 与 ABACUS 直接
> 运行的能量/力/应力精度和 wall time。所有数字来自本机实际运行。

## 0. 环境与输入

| 项 | 值 |
|---|---|
| ABACUS | `/root/abacus-develop/build_ci/abacus_basic_para`，`v3.11.0-beta7` |
| ABACUS 并行 | 1 MPI rank，`OMP_NUM_THREADS=1`（1 核） |
| ad-dft | JAX 0.12.0.1 + CUDA 12.9，RTX 3070 Ti 8GB，FP64 |
| 赝势 | `Si.pz-vbc.UPF`，sha256 `6ab22ac0…d699b` |
| 案例 | `tests/regress/si-gamma`、`tests/regress/si-distorted` |
| 物理参数 | `ecutwfc=20 Ry`（另加 40 Ry 扫描），Γ 点，`nbands=6`，`fixed`，`symmetry 0`，`cal_force=1`，`cal_stress=1` |
| ad-dft 参数 | 同一案例经 `ad.setup_from_case`；`density_factor=4`，`sigma=0` |

ABACUS 的 wall time 包含 SCF + 力 + 应力；ad-dft 的 `energy_and_forces` 和
`stress` 各自包含一次 SCF，因此**不要直接把三者相加**。最可比的是
`scf` 单列。

## 1. 精度

| 案例 | dE (meV/atom) | max dF (meV/Å) | max dσ (kbar) |
|---|---:|---:|---:|
| Si/Γ, 20 Ry | **-0.277** | 0.029 | 0.029 |
| Si 畸变, 20 Ry | **-0.293** | **0.593** | 0.028 |
| Si/Γ, 40 Ry | **+0.031** | 0.007 | 0.003 |

结论：与仓库 M0 报告一致，精度远优于门槛（E <10 meV/atom、F <10 meV/Å、
σ <5 kbar）。40 Ry 下能量差仅 0.031 meV/atom。

## 2. 性能

### Si/Γ，20 Ry，`nG=411`，`nfft=13824`

| 量 | ABACUS (1 核) | ad-dft (GPU) | 比值 |
|---|---:|---:|---:|
| SCF | 0.184 s | 0.432 s | 2.3× |
| SCF + 力 | 0.184 s（ABACUS 单次含力） | 1.35 s（`energy_and_forces`，含一次 SCF） | 7.3× |
| SCF + 应力 | 0.184 s（同上） | 1.36 s（`stress`，含一次 SCF） | 7.4× |

### Si 畸变，20 Ry，`nG=411`，`nfft=13824`

| 量 | ABACUS (1 核) | ad-dft (GPU) | 比值 |
|---|---:|---:|---:|
| SCF | 0.190 s | 0.620 s | 3.3× |
| SCF + 力 | 0.190 s | 1.45 s | 7.6× |
| SCF + 应力 | 0.190 s | 1.50 s | 7.9× |

### Si/Γ，40 Ry，`nG=1139`，`nfft=32768`

| 量 | ABACUS (1 核) | ad-dft (GPU) | 比值 |
|---|---:|---:|---:|
| SCF | 0.35 s | 1.53 s | 4.4× |
| SCF + 力 | 0.35 s | 2.38 s | 6.8× |
| SCF + 应力 | 0.35 s | 2.40 s | 6.9× |

ABACUS 在 20 核 OpenMP 下反而明显变慢（13–18 s），因此这里用 1 核作为
参考；这不是 ad-dft 的性能问题，而是该 ABACUS 构建的线程/ELPA 行为。

## 3. 差距从哪来

1. **SCF 迭代数**：ABACUS 9 次；ad-dft 设备端自适应 Broyden 为
   9 次（Si/Γ）、17 次（畸变）、9 次（40 Ry）。自适应策略是前 8 步
   `beta=0.9`、之后 `beta=0.6`；Si/Γ 与 40 Ry 已与 ABACUS 同迭代数。
2. **每步本征求解**：ad-dft 用稠密 `nG×nG` eigensolve；`nG=411` 约
   22 ms/步，`nG=1139` 约 100 ms/步。原生 G-shell scatter/gather 已把
   Hartree 从 12–19 ms 降到 ~1.5 ms、density 从 7.6–18.6 ms 降到
   ~0.13 ms，因此 eigh 现在是单步的主要成本。
3. **小体系**：`nfft` 只有 1.4–3.3 万，GPU launch/同步开销占比不低；
   ABACUS 在本机 1 核上非常快。

## 4. 复现

```bash
# ABACUS
cd /tmp/abacus_si_gamma
OMP_NUM_THREADS=1 /root/abacus-develop/build_ci/abacus_basic_para

# ad-dft
export PYTHONPATH=src:tests:.
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export ADDFT_NATIVE_XC=1 ADDFT_NATIVE_EIGH=1
python - <<'PY'
import time, jax, addft as ad
s = ad.setup_from_case("tests/regress/si-gamma")
st = ad.scf(s); jax.block_until_ready(st.rho_r)
t = time.perf_counter(); st = ad.scf(s); jax.block_until_ready(st.rho_r)
print("scf", time.perf_counter() - t, int(st.n_iter))
PY
```

## 5. 下一步

- Davidson/LOBPCG + 上一步波函数 warm start：大 `nG` 的主要杠杆。
  （已试 Krylov-Ritz/LOBPCG；在当前 411–1139 规模上仍不如 dense eigh，
  且秩亏子空间不稳定，暂未合入。）
- Broyden 的线搜索/信赖域：允许更高 `beta` 而不在畸变构型发散。
- 原子密度初始猜测，减少早期迭代。
- 进一步融合 Hartree/density 的 FFT epilogue。

---

## 6. 100 Ry 对照

同一 Si/Γ 案例，`ecutwfc=100 Ry`（ad-dft `ecut=50 Ha`），
`nG=4477`、`nG_rho=35749`、`nfft=110592`、`nband=6`。

### 6.1 精度

| 量 | ABACUS | ad-dft | 差 |
|---|---:|---:|---:|
| `FINAL_ETOT_IS` (eV) | -198.3550491872 | -198.3549781205 | **+0.0355 meV/atom** |
| max force (meV/Å) | ~0 | ~0 | **0.0092 meV/Å** |
| stress (kbar) | 422.45993 | 422.45969 | **0.0004 kbar** |

### 6.2 总体性能

两者都是 **10 次电子迭代**。

| 量 | ABACUS 1 核 | ad-dft GPU | 比值 |
|---|---:|---:|---:|
| SCF wall | **0.68 s** | **~1.2 s** | **1.8×** |

ad-dft 首次调用含编译约 41.4 s；35.2 s 是 warm 结果。
ABACUS 内部计时总计 0.42 s（wall 0.68 s）。

### 6.3 ad-dft 分量（100 Ry）

| 步骤 | 单次 | 调用数 | 总计 | 占 SCF |
|---|---:|---:|---:|---:|
| **Davidson（matrix-free）** | **~0.04–0.09 s/次** | 14 | **~1.3 s** | **~96%** |
| Hψ（6 bands） | 1.59 ms | 14 | 0.022 s | 1% |
| effective potential | 1.49 ms | 14 | 0.021 s | 1% |
| density | 1.40 ms | 14 | 0.020 s | 1% |
| XC | 2 ms | 14 | 0.028 s | 1% |
| occupations | <1 ms | 15 | ~0 | — |
| geometry cache | 0.13 s | 1 | 0.13 s | 0.4% |

15 次 eigh = 10 次主循环 + 4 次 polish + 1 次最终本征求解。
杂化/残差等合计约 0.1 s。

### 6.4 ABACUS 分量（100 Ry，内部计时）

| 项 | 时间 | 占比 | 说明 |
|---|---:|---:|---|
| `hamilt2rho_single` | 0.24 s | 57.6% | 一次电子步 |
| `HSolverPW solve` | 0.24 s | 57.6% | 含 Davidson |
| `hPsi` | 0.17 s | 41.6% | 34 次 matvec |
| `veff_pw` | 0.17 s | 39.9% | 势作用/FFT |
| `recip2real` | 0.11 s | 26.8% | 220 次 |
| `cal_grad` | 0.11 s | 26.5% | Davidson 残差/预条件 |
| `real2recip` | 0.07 s | 17.1% | 160 次 |
| `diag_zhegvx` | 0.01 s | **1.6%** | 小空间对角化 |
| `cal_veff` | 0.05 s | 11.4% | 含 Hartree/XC |
| `psiToRho` | 0.03 s | 8.3% | 密度 |
| `mix_rho` | 0.03 s | 6.5% | Broyden |
| force + stress | 0.02 s | ~3.9% | — |

> ABACUS timer 是嵌套的，百分比会重叠；重点是数量级：
> 它的 `diag_zhegvx` 只占 1.6%，而 ad-dft 的 dense eigh 占 99%。

### 6.5 结论

100 Ry 下 **双方迭代数相同（10 次）**，但 Davidson 内部实现和
matvec 数不同：

- ABACUS：34 次 `hPsi` + 34 次小空间 `diag_zhegvx`，合计 0.43 s；
- ad-dft：约 1 cold 15 + 9 warm 7 + 4 polish 7，每次 Hψ 只作用在
  新列上（Hψ 缓存）；单次 6-band Hψ 1.59 ms，等效 matvec 约 500–600 次。

随截断升高的 scaling：

| `ecutwfc` | `nG` | ad-dft SCF | ABACUS | 比值 |
|---|---:|---:|---:|---:|
| 20 Ry | 411 | 0.45 s | 0.184 s | 2.4× |
| 40 Ry | 1139 | 1.53 s | 0.35 s | 4.4× |
| 100 Ry | 4477 | ~1.2 s（Davidson） | 0.68 s | **1.8×** |

下一步不是再写一个 Davidson，而是减少 Davidson 内部 matvec：
RMM-DIIS/更强预条件/更小 warm 子空间，以及 FP32/fused Hψ。


---

## 7. 公平基线补充（100 Ry）

| 平台 | wall / 说明 |
|---|---:|
| ABACUS CPU 1 核 | 0.67 s（含 force/stress） |
| ABACUS MPI 4 核 | **0.35 s** |
| ABACUS MPI 8 核 | 0.42 s |
| ABACUS MPI 20 核 | 2.50 s（该构建线程/ELPA 配置不佳） |
| ABACUS GPU（v3.10.1） | 2.51 s（含 force/stress） |
| ad-dft GPU | 1.23 s SCF / 2.33 s E+F / 2.24 s stress |
| ad-dft CPU 1 线程 | **10.97 s SCF** |

注意：ABACUS GPU 是 v3.10.1，CPU 是 v3.11.0-beta7，版本不同；GPU 基线需要
同 commit GPU 构建后再最终定论。

当前结论：

- ad-dft GPU 相对 ABACUS 1 核：SCF 约 1.8× 慢，E+F 约 3.4× 慢；
- 相对 ABACUS 4 核：E+F 约 6.7× 慢；
- ad-dft GPU 相对 ad-dft CPU：约 9× 加速；
- 因此“GPU 打 CPU”不能证明 10×；必须继续算法/表示优化。


## 8. 16 原子（20 Ry）补充

| 平台 | SCF |
|---|---:|
| ABACUS CPU 1 核 | 1.86 s |
| ABACUS MPI 4 | 0.79 s |
| ABACUS MPI 8 | 0.76 s |
| ad-dft GPU（auto→Davidson） | 5.2 s（32 iters） |

当前 ad-dft 比 ABACUS 8 MPI 慢约 6.8×；主要差距是 SCF 迭代数（32 vs 9）
和单步 Davidson/QR/FFT 成本，而不是 dense eigh。
