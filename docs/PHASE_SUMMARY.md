# 阶段总结与模块开发档案

> 用途：记录各模块的设计考虑、尝试过的方案、实测效果和当前状态，供后续开发查阅。  
> 环境基线：RTX 3070 Ti 8GB / WSL2，JAX 0.12.0.1 + CUDA 12.9，FP64；
> ABACUS v3.11.0-beta7，1 MPI / 1 OpenMP 线程。  
> 详细实验：`docs/experiments/`；性能路线：`docs/DEV_PLAN_10X.md`；
> ABACUS 对照：`docs/reports/ABACUS_COMPARISON.md`。

---

## 0. 一页速览

### 最近进展（2026-10-29）

- `nG>3000` 自动使用 matrix-free Davidson；
- `max_sub=max(48, 4*nband)`，修复 16 原子 `nband=34` 子空间过小的问题；
- 100 Ry SCF ~1.2 s；16 原子 20 Ry SCF 5.2 s（dense 43.2 s）；
- 公平基线：ABACUS 1 核 0.67 s、4 MPI 0.35 s、8 MPI 0.76 s（16 原子）；
- 自演化：`max_sub=48, cold=15, warm=5, ABACUS 预条件 α=1.0, Broyden 0.9→0.5`；
- 全量非 slow + slow 力测试通过；唯一失败是外部 ABACUS 示例路径缺失。


### 项目定位

ad-dft 是一个**可微周期性平面波 DFT 内核**：能量是唯一可微标量，
力/应力/极化/约束乘子是其导数；SCF 用不动点隐式微分。

### 当前状态

| 层 / 模块 | 状态 | 说明 |
|---|---|---|
| L0 运行时（单位/dtype） | 稳定 | Ry↔Ha 只在边界转换；x64 |
| L1 几何（Cell/Basis/KMesh/UPF） | 稳定 | 可微 Cell；冻结壳层；FFT 盒规则 |
| L2 原语（密度/Hartree/XC/local/nonlocal/Ewald/occupation） | 稳定 | LDA；单 species；NLCC 显式拒绝 |
| L3 求解器（eigh/mixer/SCF） | 可用 | 小 nG dense eigh；nG>3000 matrix-free Davidson；Broyden/Kerker |
| L4 微分（envelope/implicit） | 稳定 | I-15、FD 力、伴随残差门禁通过 |
| L5 观测量（能量/力/应力/ΔQ） | 稳定 | 与 ABACUS 对照通过 |
| 原生 XC kernel | 稳定 | 18.8 ms → 0.18 ms |
| 原生 cuSOLVER eigh | 可用 | 小 `nG` 与 XLA 持平，大 `nG` 约 1.3× |
| 原生 pack scatter/gather | 稳定 | complex128/64；`real_from_g` 7.3/29.8 ms → 0.2/1.34 ms |
| Matrix-free Hψ | 生产（nG>3000） | 已接入 Davidson；100 Ry 1.26 ms（FP64） |
| Davidson | 生产（nG>3000） | 100 Ry ~1.2s；16 原子 ~5.2s；小 nG 保留 dense |
| LOBPCG/压缩基 | 未达生产 | 见 E2a/E5 |

### 当前性能基线（SCF）

| 案例 | `nG` | `nfft` | ABACUS 1 核 | ad-dft GPU | 比值 |
|---|---:|---:|---:|---:|---:|
| Si/Γ 20 Ry | 411 | 13824 | 0.184 s | **0.45 s** | 2.4× |
| Si/Γ 40 Ry | 1139 | 32768 | 0.35 s | **1.53 s** | 4.4× |
| Si/Γ 100 Ry | 4477 | 110592 | 0.68 s（1 核） | **~1.2 s** | 1.8× |
| 16 原子 20 Ry | 3143 | 91125 | 1.86 s（1 核） | **5.2 s** | 2.8× |

16 原子 ABACUS 8 MPI 为 0.76 s，因此 ad-dft 相对 8 MPI 约 6.8× 慢。

公平基线：ABACUS MPI 4 核 0.35 s；ad-dft GPU E+F 2.33 s；ad-dft CPU 1 线程
SCF 10.97 s。16 原子 20 Ry：ABACUS 8 MPI 0.76 s，ad-dft GPU 5.2 s。
当前 GPU-vs-1CPU 对比不能证明 10×。

当前 E+F/σ（warm，value_and_grad 已改为单次 SCF）：

| 案例 | `energy_and_forces` | `stress` |
|---|---:|---:|
| Si/Γ 20 Ry | 1.35 s | 1.36 s |
| Si 畸变 20 Ry | 1.45 s | 1.50 s |
| Si/Γ 40 Ry | 2.38 s | 2.40 s |
| Si/Γ 100 Ry | 2.33 s | 2.24 s |

100 Ry 的 dense-eigh 路径是 35.4 s；`nG>3000` 已切到 matrix-free
Davidson，SCF ~1.2 s。16 原子 20 Ry 修好 `max_sub=max(48,4*nband)` 后
从 dense 43.2 s 降到 5.2 s；主要成本是 Davidson matvec/QR，而不是 FFT/XC。

### 当前生产路径

```
Cell/Basis (冻结壳层)
  → geometry cache (v_loc, projectors, D)
  → effective potential (Hartree + XC + local)
  → nG>3000: matrix-free Davidson + cached Hψ
     否则: dense H + native cuSOLVER eigh_lowest
  → occupations (JIT)
  → device-side Broyden SCF (lax.while_loop)
  → 4 步 fixed-point polish
  → envelope theorem: 力/应力
```

可关闭的 native 路径：

```bash
ADDFT_NATIVE_XC=0     # 关闭原生 XC
ADDFT_NATIVE_EIGH=0   # 关闭原生 eigh
ADDFT_NATIVE_PACK=0   # 关闭原生 scatter/gather
```

也可 `driver="python"` 强制旧 SCF 路径。

---

## 1. 设计原则与不可妥协约束

七条设计原则：

1. AD-native 但不是 AD-only：每个原语有 ref/JAX/native 三份实现。
2. periodic-first：`Cell` 是可微叶子；`KMesh` 权重和为 1；投影随几何重建。
3. 收敛与微分解耦：L4 只依赖收敛态，不依赖迭代历史。
4. 观测量是能量的导数，不是独立模块。
5. 性能关键路径下沉，物理组合上浮。
6. 可验证性优先于性能：梯度门禁与性能门禁并列。
7. 契约先行：离散化/泛函/赝势/单位/边界条件改动先改 `spec/`。

不可妥协：

- 同一赝势、同一 `ecutwfc`、同一 k 点、同一 `nband`、同一占据；
- E/F/σ 与 ABACUS 的差值不劣于现有门禁；
- I-15、FD 力/应力、ΔQ 伴随门禁不退化；
- JAX 参考路径始终保留；
- native 后端可关闭、可回退；
- 不静默换基组/换赝势。

---

## 2. 架构与数据流

```
runtime (units/dtypes)
  │
  ▼
geometry (Cell, Basis, KMesh, Pseudo/UPF)
  │
  ▼
physics primitives
  density / Hartree / XC / local / nonlocal / Ewald / occupation
  │
  ▼
solver
  geometry cache → effective_potential → H → eigh → occupations → mixer/SCF
  │
  ▼
diff
  envelope theorem / fixed-point implicit differentiation / response transpose
  │
  ▼
observables
  energy decomposition / forces / stress / ΔQ constraint
```

native 层旁路：

```
native/xc_lda.cu   → JAX FFI → physics.xc
native/eigh.cu     → JAX FFI → solver.eigh
native/pack.cu     → JAX FFI → geometry.basis
native/hpsi.cu     → (计划)  → ops.hpsi
```

---

## 3. 模块档案

### L0 运行时

**文件**：`src/addft/runtime/units.py`、`dtypes.py`

**设计考虑**

- ABACUS 内部 Ry–Bohr；ad-dft 内部 Ha–Bohr；
- Ry→Ha 只在 adapter/UPF 入口做一次，`src/` 内不出现 `*_to_ry`；
- 统一 `jax_enable_x64=True`；
- `AccuracyLedger` 记录精度/误差。

**尝试/结论**

- FP64 路径稳定；FP32 eigh 实验见 E4，未进入主路径；
- native kernel 以 FP64 为正确性基线。

**门禁**：单位换算、UPF 解析、能量分解。

---

### L1 几何

**文件**：`src/addft/geometry/{cell,basis,kmesh,psp,upf}.py`

**设计考虑**

- `Cell.lattice/frac` 是唯一可微叶子；不在 L1 转 numpy；
- `Basis.miller` 在参考构型冻结；`gvectors` 每步从 lattice 重算，保留 Pulay；
- FFT 盒规则与 ABACUS 一致：奇数 → 2·3·5 光滑数；
- 局域势/密度使用 `miller_rho`（`ecutrho = density_factor × ecutwfc`）；
- UPF v1/v2；`D_ij` 0-based、`l²+m` 展平；NLCC 显式拒绝；
- 单 species（R3 限制）。

**尝试/结论**

| 尝试 | 结果 |
|---|---|
| 合成赝势 vs 真实 UPF | 都支持 |
| 原子密度初始猜测 | 未减少 Si 的 SCF 迭代，放弃 |
| JAX scatter/gather | 慢：`real_from_g` 7.3–29.8 ms |
| 原生 pack scatter/gather | 成功：0.2–1.34 ms；见 E1/native pack |
| 压缩/自适应基组 | 未实现；E5 parked |
| 分而治之 | 未实现；E7 大体系 track |

**当前限制**

- 单 species；
- 非 Γ 使用联合壳层（比 per-k 略大）；
- 壳层冻结要求 `shell_margin > 5%`；
- 压缩基组/PAW/NAO 不在主路径。

**测试**：`test_contracts.py`、`test_kpoints.py`、`test_upf.py`、
`test_ylm.py`、`test_native_pack.py`。

---

### L2 物理原语

#### 密度 `physics/density.py`

**考虑**

- `n(r)=Σ w_k f_nk g_s |ψ_nk(r)|²`；
- k 点相位不可比，必须先各自取模方；
- 权重归一化、自旋简并因子是高频错误源。

**尝试/效果**

- JAX FFT + scatter 路径；
- native pack 后 density 0.13 ms（100 Ry）；
- 密度 FFT 本身不是瓶颈。

#### Hartree `physics/hartree.py`

**考虑**

- `V_H(G)=4πn(G)/G²`，G=0 置零（中性背景）；
- 线性算子，VJP 即伴随；
- 能量与势必须同源。

**尝试/效果**

| 版本 | 20 Ry | 40 Ry | 100 Ry |
|---|---:|---:|---:|
| JAX scatter/gather | 12.0 ms | 19.1 ms | — |
| native pack | **1.5 ms** | **1.5 ms** | **~2 ms** |

#### XC `physics/xc.py`

**考虑**

- 只给能量密度，势由 AD 生成；
- 离散泛函导数权重 `(Ω/N)` 是契约；
- libxc 通过 `pure_callback`，不能进 jit 循环；
- 原生 LDA 融合 kernel 同时给 `E_xc` 与 `v_xc`。

**尝试/效果**

| 方案 | 结果 |
|---|---|
| libxc host callback | 18.8 ms/call |
| 原生 fused LDA XC | **0.18 ms/call** |
| 原生 XC 的 custom_vjp | fxc 回退 libxc/analysis，正确 |
| 设备 SCF 内 libxc | 不可 jit；自动回退 Python driver |
| FP32 XC | 未单独尝试；当前 XC 已不是瓶颈 |

**限制**：LDA/PZ81；GGA 未 native 化；libxc 仍是参考。

#### 局域势 `physics/local.py`

**考虑**

- 结构因子 `e^{-iG·R}` 与 `1/Ω` 必须在图里；
- V_loc 必须在密度壳层上采样，否则矩阵元被截断；
- 每步重建很贵 → geometry cache。

**尝试/效果**

- 原始 `local_potential_r` 80 ms（100 Ry，首次）；
- geometry cache 后 SCF 内 0 ms（只读缓存）；
- cache 首次 ~5 s（含编译），warm ~0.13 s。

#### 非局域 `physics/nonlocal_op.py`

**考虑**

- `β_Ii(G)` 从几何生成；径向插值必须可微；
- `D_ij` 索引 `(i,p,j,q)`，避免块交错；
- `1/√Ω` 指数必须是 −1/2（应力体积项）；
- 每步重建 projectors/D 很贵 → cache。

**尝试/效果**

- `projectors` ~41–74 ms/次；`dion_full` ~44–47 ms/次；
- geometry cache 后 SCF 内只读；
- 非局域应用仍是 `vnl_apply` 两次 GEMM；未做 fused triple product。

#### Ewald `physics/ewald.py`

**考虑**

- real + reciprocal + self + background；
- η 无关性、平移不变；
- 几何相关，SCF 内固定 → 可缓存。

**尝试/结论**

- 当前实现含 `g[keep]` 布尔索引，不能直接进 `jax.jit`；
- 在直接最小化原型中预先计算 Ewald 常数，避免 jit 问题；
- 后续可将 Ewald 常量放进 geometry cache。

#### 占据 `physics/occupation.py`

**考虑**

- Fermi–Dirac；μ 由电子数守恒隐式确定；
- σ→0 走整数占据，不能把 σ 调小硬撑；
- 力必须来自 Mermin 自由能 `F=E−σS`。

**尝试/效果**

| 方案 | 结果 |
|---|---|
| eager 80 次二分 | ~70 ms |
| `jax.jit` + static sigma/spin | **~0.75 ms** |
| 更长历史/更少迭代 | 未采用；jit 已足够 |

**限制**：σ 作为静态参数；不支持 σ 的动态 AD。

---

### L3 求解器

#### eigh `solver/eigh.py`

**考虑**

- 只接受子空间不变输出（密度矩阵/投影/总能量）；
- 禁止单本征值求导；
- 小 nG 用 dense eigh；nG>3000 用 matrix-free Davidson；max_sub=max(48,4*nband)，eig_tol=1e-3。

**尝试/效果**

| 方案 | 结果 |
|---|---|
| `jnp.linalg.eigh` | 20 Ry 22 ms；100 Ry 2.36 s |
| 原生 cuSOLVER `zheevdx` range=I | 小 `nG` 持平；`nG=1139` 约 1.3× |
| LOBPCG warm | 100 Ry 5 步残差 ~8e-3 |
| Chebyshev 滤波 | 100 Ry degree 20 残差 ~9e-3 |
| block Krylov | 100 Ry m=16 残差 ~2.9e-3 |
| FP32 eigh | 3.5× 快，但本征值差 ~3e-6 Ha |
| FP32 + `orth(HψX)` | 收敛到高态，失败 |

**结论**：dense eigh 仍是 100 Ry 主瓶颈；需要 Lanczos/shift-invert/
强预条件，或压缩基组。

#### mixer `solver/mixer.py`

**考虑**

- 混合器是加速器，不进微分图；
- DIIS 系数由历史残差决定；
- 历史固定，避免动态 shape。

**尝试/效果**

| 方案 | 结果 |
|---|---|
| eager DIIS | ~350 ms/次 |
| JIT DIIS | ~3 ms/次 |
| Kerker + DIIS residual | 迭代反而增加（46–100 次） |
| simplified Broyden | 31 → 15/17 次（20 Ry） |
| 自适应 Broyden 0.9→0.6 | Si/Γ 9 次、40 Ry 9 次、畸变 17 次 |
| Broyden beta=0.7+ 畸变 | 发散/停滞；保留 0.6 |
| Python DIIS fallback | 保留 |

#### SCF `solver/scf.py`

**考虑**

- 正向 SCF 与微分解耦；
- 设备端 `lax.while_loop` 去掉 Python 循环/host sync；
- 4 步 raw fixed-point polish（包络定理要求严格不动点）；
- 不收敛回退 Python DIIS；
- libxc 不能进 jit 循环 → 自动回退。

**尝试/效果**

| 版本 | Si/Γ 20 Ry | 畸变 20 Ry | 100 Ry |
|---|---:|---:|---:|
| Python DIIS | 1.48 s | 1.87 s | — |
| 设备 Kerker | 0.72 s | 1.13 s | — |
| 设备 DIIS | 0.88 s | 0.97 s | — |
| 设备自适应 Broyden | **0.45 s** | **0.66 s** | **35.4 s** |

#### geometry cache `solver/geometry_cache.py`

**考虑**

- `v_loc`、`projectors`、`D` 只依赖几何；
- SCF 内固定；结构优化每步重建；
- JVP 路径不能复用旧几何缓存。

**效果**

- 去掉每步 ~134 ms 几何重建（合成体系）；
- warm 100 Ry cache ~0.13 s；首次含编译 ~5 s。

---

### L4 微分层

**文件**：`diff/implicit.py`、`physics/response.py`、`api.py`

**设计考虑**

- 能量：envelope theorem；冻结 ψ，ρ 随几何重建；
- 非能量观测量：`(I−∂F/∂n)^T λ = ∂L/∂n`，matrix-free/BiCGSTAB；
- 不展开 SCF 迭代；
- 简并处用占据加权极限，不产生 NaN。

**尝试/效果**

- I-15 在 `nfft=3375`：`corr≈1`、`rel=1.1e-6`；
- 原生 eigh 相位问题：`make_response` 用 `<C_occ|ψ>` 酉极因子对齐；
- native pack：`linear_transpose` 不能转置 FFI → response/adjoint 临时回退 JAX；
- 原生 XC custom_vjp；eigh 只做 primal；
- ΔQ 伴随残差 ~2e-11。

**限制**

- response/adjoint 路径仍用 JAX fallback，未 native 化；
- 伴随 BiCGSTAB 成本约 2–3× SCF。

---

### L5 观测量

**文件**：`observables/energy.py`、`constraint.py`

**考虑**

- 变分形式 `E = Σf<ψ|H|ψ> − E_H + E_xc − ∫v_xc n + E_II`；
- 力/应力来自包络定理，不单独写 Hellmann–Feynman 代码；
- ΔQ 必须走伴随，因为不是驻点泛函。

**效果**

- 与 ABACUS：20 Ry E 差 -0.277 meV/atom，F 0.593 meV/Å，σ 0.029 kbar；
- 40 Ry E +0.031 meV/atom，F 0.007 meV/Å，σ 0.003 kbar；
- 100 Ry E +0.0355 meV/atom，F 0.0092 meV/Å，σ 0.0004 kbar。

---

## 4. 原生 kernel 档案

### `native/xc_lda.cu`

- fused LDA XC：一次 pass 算 `E_xc`、`v_xc`；
- FFI + `custom_vjp`（fxc 回退）；
- 18.8 ms → 0.18 ms；
- 失败点：无 GPU/未构建时回退；libxc 路径保留。

### `native/eigh.cu`

- cuSOLVER `zheevdx` range=I，只算 lowest `nband`；
- 输入 row-major → column-major 转置；
- primal-only；微分层用微扰；
- 需要注意 cuSOLVER 的任意相位；response 做子空间对齐；
- 小 `nG` 持平；`nG=1139` 约 1.3×；100 Ry 仍是 2.36 s。

### `native/pack.cu`

- complex128/complex64 scatter/gather；
- forward 原生，JVP/VJP 用 JAX fallback（`custom_jvp`）；
- response/adjoint 临时关闭 native；
- `real_from_g` 7.3/29.8 ms → 0.2/1.34 ms；
- FP32 Hψ 约 1.5 ms（FP64 ~3.7 ms），但仍 >0.5 ms 目标；
- 失败点：直接 FFI 不可微；`linear_transpose` 会绕过 custom_jvp 转置原始 FFI。

### `src/addft/ops/hpsi.py`（实验）

- matrix-free `Hψ`：T + V + V_NL；
- 100 Ry 1.26 ms；dense Hψ 707 ms；
- 已接入 nG>4000 的 Davidson 主路径；直接最小化仍待开发。

### 构建与开关

```bash
NVCC=/root/miniconda/bin/nvcc ./native/build.sh   # CUDA >= 12
```

| 开关 | 作用 |
|---|---|
| `ADDFT_NATIVE_XC=0` | 关闭原生 XC |
| `ADDFT_NATIVE_EIGH=0` | 关闭原生 eigh |
| `ADDFT_NATIVE_PACK=0` | 关闭原生 pack |
| `driver="python"` | 强制 Python SCF |
| `driver="device"` | 强制设备 SCF |

---

## 5. 性能实验台账（E0–E6）

| ID | 方案 | 状态 | 关键数据 | 决策 |
|---|---|---|---|---|
| E0 | ABACUS 基线 | green | 20/40/100 Ry | 继续 |
| E1 | Matrix-free Hψ | green* | 1.26 ms @100 Ry（FP64） | 已落地；FP32 待做 |
| E2 | LOBPCG/Chebyshev/Krylov | yellow | 残差 ~1e-3–9e-3 | 不进入主路径 |
| E3 | 直接最小化能量 | yellow | 0.1 mHa 后停滞 | 需 L-BFGS |
| E4 | FP32 eigh/混合精度 | yellow | 3.5×，误差 3e-6 Ha | 需精化 |
| E5 | 压缩/自适应基组 | parked | 未实现 | 独立 track |
| E6 | 密度矩阵纯化 | yellow | 20 Ry 1e-14；100 Ry FP64 太慢 | 仅大体系 |
| E7 | 分而治之 | parked | 未实现 | 大体系 |
| E8 | Persistent/CUDA Graph | partial | device loop/native pack | 继续 |
| E9/E10 | PAW/NAO、TT/神经 | parked | 未实现 | 契约/研究 |

详细记录：

- `docs/experiments/E0-baseline.md`
- `E1-hpsi.md`
- `E2-lobpcg.md`
- `E3-direct-min.md`
- `E4-fp32-eigh.md`
- `E6-density-matrix.md`

---

## 6. 失败模式与“不要重复”

1. **不要用 JAX 通用 scatter/gather 做 G 打包**：100 Ry 的
   `real_from_g` 29.8 ms；native pack 1.34 ms。
2. **不要直接对 FFI scatter/gather 求导**：JAX 会报不可微；
   forward 用 native、JVP/VJP 用 JAX fallback。
3. **不要在 `linear_transpose` 路径里使用未加保护的 native pack**：
   response/adjoint 要临时关闭 native。
4. **不要对 SCF 迭代展开求导**：用 envelope/implicit；展开只会增加显存。
5. **不要盲目提高 Broyden beta**：0.7+ 在畸变 Si 上发散/停滞；
   用 0.9→0.6 自适应 + DIIS fallback。
6. **Kerker 预条件不是万能的**：在当前 DIIS 形式上反而增加迭代。
7. **不要用 `X ← orth(HX)` 精化最低态**：会收敛到高态。
8. **不要只靠 FP32 eigh 交付精度**：本征值差 ~3e-6 Ha，力/应力会受影响。
9. **不要在 libxc host callback 上做 jit**：设备 SCF 要 native XC 或 analytic。
10. **不要在小体系上强推 direct minimization**：朴素梯度会停滞，
    需要 Riemannian L-BFGS/trust-region。

---

## 7. 验证与测试

### 关键门禁

- 总能 vs ABACUS；
- 力 vs ABACUS / FD；
- 应力 vs ABACUS / FD；
- I-15 响应算子 vs `scf_map` FD；
- ΔQ 伴随 vs FD；
- 原生 XC/eigh/pack 与 JAX 对照；
- 设备 SCF vs Python DIIS。

### 当前测试状态

```text
test_abacus_io.py            1 failed, 4 passed   # 外部 /root/abacus-develop 示例缺失
test_contracts.py            6 passed
test_forces.py               3 passed, 5 deselected
test_harness_catches_bugs.py 6 passed
test_hpsi.py                 3 passed
test_invariants.py           8 passed
test_kpoints.py              3 passed
test_libxc.py                2 passed
test_native_eigh.py          3 passed
test_native_pack.py          3 passed
test_native_xc.py            2 passed
test_oracles.py              2 passed
test_davidson.py             1 passed
test_primitives.py           13 passed
test_scf_driver.py           2 passed
test_upf.py                  2 passed, 1 skipped
test_ylm.py                  3 passed
slow forces                  5 passed
```

### 复现

```bash
export PYTHONPATH=src:tests:.
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export ADDFT_NATIVE_XC=1 ADDFT_NATIVE_EIGH=1 ADDFT_NATIVE_PACK=1
bash run_tests.sh
python -m pytest tests/test_forces.py -m slow
```

---

## 8. 已知缺口与下一步

### 当前最大瓶颈

- 100 Ry：Davidson 已替代 dense eigh，SCF 1.35 s；瓶颈变为 Davidson 内部 matvec/QR 数；
- 20/40 Ry：dense eigh 仍是单步主成本；
- native XC/pack/Hartree/density 已不是瓶颈。

### 下一步优先级

1. **E2b：Lanczos/shift-invert + 强预条件**  
   目标：<20 matvec 到 1e-8；避免 dense eigh。
2. **E3b：Riemannian L-BFGS/trust-region**  
   目标：<15 次能量/梯度评估；SCF-free。
3. **E5：压缩/自适应基组**  
   需契约决策；可能直接改变 10× 量级。
4. **E7/E6：大体系 linear scaling**  
   只在大体系启用。
5. **E8：CUDA Graph/persistent kernel**  
   在算法稳定后做。

### 当前不做

- 不再微调 dense eigh；
- 不把未过门禁的 LOBPCG/压缩基放进主路径；
- 不在同 PP 对照中混入换基组/换赝势。

---

## 9. 文件地图（开发速查）

| 路径 | 内容 |
|---|---|
| `spec/` | 物理/离散化契约 |
| `src/addft/runtime/` | 单位、dtype、精度账本 |
| `src/addft/geometry/` | Cell/Basis/KMesh/Pseudo/UPF |
| `src/addft/physics/` | 密度/Hartree/XC/local/nonlocal/Ewald/occupation/response |
| `src/addft/solver/` | eigh/mixer/SCF/geometry cache |
| `src/addft/diff/` | 不动点隐式微分 |
| `src/addft/observables/` | 能量/力/应力/约束 |
| `src/addft/kernels/` | native FFI 加载器 |
| `src/addft/ops/hpsi.py` | matrix-free Hψ（实验） |
| `native/` | CUDA kernels + build.sh |
| `tests/` | 单元/回归/门禁 |
| `bench/` | 性能基准；`bench/dashboard.py` 生成实时 HTML/JSON 看板 |
| `docs/DEV_PLAN_10X.md` | 10× 路线与实验台账 |
| `docs/experiments/` | 逐项实验记录 |
| `docs/reports/ABACUS_COMPARISON.md` | 20/40/100 Ry 对照 |
| `docs/reports/M0.md` / `M1.md` | 阶段出口报告 |


---

## 10. Review 修复记录（本轮）

| # | 问题 | 修复 |
|---:|---|---|
| 1 | `energy_and_forces` 先 `e_fn` 再 `jax.grad(e_fn)`，SCF 跑两次 | 改成 `jax.value_and_grad(e_fn)`，E+F 20 Ry 2.19→1.35 s |
| 2 | `ewald_energy` 用布尔索引，不能进 `jax.jit` | 改成 mask + `jnp.where`；新增 `test_ewald_is_jittable` |
| 3 | `pack.py` 有不可用的死函数 `scatter()` | 删除 |
| 4 | `implicit_grad` 异常时 native pack flag 不恢复 | 改为 `with native_pack(False)` + try/finally |
| 5 | `response.make_response` 手写 flag 恢复 | 改用 context manager |
| 6 | `DEV_PLAN` E1 写成 `<1ms` 但状态 green | 改为 `FP64 <2ms / FP32 <1ms`，状态 `green*` |
| 7 | `PHASE_SUMMARY` E1 状态与 gate 不一致 | 同步为 `green*`，注明 FP32 待做 |
| 8 | `M1.md` 前半部分是 native pack 前快照 | 加“历史快照”说明，指向第 7 节和最新对照 |
| 9 | ABACUS 对照的 E+F/σ 数字过期 | 更新为 20/40 Ry 当前值（1.35/2.38 s 等） |
| 10 | 100 Ry 分量表 Hartree/density 仍是旧值 | 更新为 native pack 后 1.5 ms / 0.13 ms |
| 11 | `README` 仍说“第一个原生 kernel” | 更新为 XC + eigh + pack |
| 12 | 阶段总结缺 E+F/σ 当前基线 | 已补 20/40/100 Ry 表 |

**验证：** 全量非 slow + slow 力测试通过；唯一失败仍是外部
`/root/abacus-develop/examples/...` 路径缺失的 `test_abacus_io.py`。
