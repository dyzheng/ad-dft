# 契约 2：离散化（壳层、网格、表示）

## 1. 壳层冻结

平面波基组 = 整数指标集合 `miller`，在**参考构型**一次性确定，之后冻结：

```python
Basis.miller : (nG, 3) int32      # 冻结；不参与微分
gvectors(basis, lattice, k)       # 每次从 lattice 重算 → 可微
```

局域势不能与波函数共用同一壳层：`⟨G|V_loc|G'⟩` 使用 `V_loc(G−G')`，
而两个波函数 G 的差可以达到 `2|G|_max`。因此 `Basis` 另存

```python
Basis.miller_rho : (nG_rho, 3) int32  # |G|²/2 ≤ density_factor * ecut
gvectors_density(basis, lattice)      # 局域势/密度壳层的笛卡尔 G
```

默认 `density_factor=4`，对应 ABACUS 的 `ecutrho = 4 × ecutwfc`。
`V_loc(G)` 在 `miller_rho` 上采样后投到共用 FFT 盒；哈密顿量仍只在
`miller` 上取矩阵元。把 `V_loc` 截到波函数壳层会静默丢掉高 G 矩阵元；
Si/Γ 实测可使总能偏差约 31 meV/atom、本征值偏差 5–12 meV。

判据 `|k+G|²/2 ≤ ecut`（原子单位）。应变下 `G → (1+ε)^{-T}G`、`Ω → det(I+ε)Ω`
自动进入 AD 图，三项 Pulay 一个不少。

**为什么冻结**：不冻结时，某些 G 会在应变中跨过截断面，能量连续但导数不连续，
AD 会在跳变点给出尖峰污染整条应力曲线。这是离散化的固有性质，不是 AD 缺陷。

验收要求：`shell_margin() > 5%`（最外层 |G|² 与截断面的相对距离），
且"壳层跳变处的应力不连续 < 0.1 kbar"。

## 2. FFT 盒规则（与 ABACUS 逐字对齐）

`source_basis/module_pw/pw_init.cpp:92-137`：先 `ibox = 2·max|m_i| + 1`（奇数），
再自增到**只含 2、3、5 因子**的最小整数。源码里 7 的分支被注释掉
（`//if (b%7==0){...}`），因此 7 不允许。

密度网格：ABACUS 默认 `ecutrho = 4 × ecutwfc`（`read_inp_sys.cpp:1227`），
波函数 FFT 复用密度网格（`setup_pwwfc.cpp:60-66`）。

**追加约束（本仓库实测得出）**：若在同一能量里混用"谱式矩阵表示"与
"实空间网格积分"表示，则 `v · |ψ|²` 的频率可达 3× 波函数截断。要求

```
π·n_i / L_i  >  3 · |G|_max        （混用表示时）
π·n_i / L_i  >  2 · |G|_max        （纯网格表示时）
```

不满足时，两种表示给出的同一个量相差可达 34%（实测），
表现为能量形式互不自洽、力偏差 20–30%。

## 3. 表示一致性（本项目的核心工程约束）

对任何一个一体项 `⟨ψ|V|ψ⟩`，只允许一种实现：

* **谱式/矩阵**：`V_GG' = ṽ(G−G')`，期望值 `Σ c* V c`；
* 或 **实空间**：`(Ω/N)Σ_j v_j |ψ_j|²`。

两者在 `V` 非带限时**不相等**（差值 = 混叠项）。混用会破坏驻点性。
约定：**一体项（T, V_loc, V_NL, V_xc 的双重计数）一律走矩阵**；
**二体与密度泛函（E_H, E_xc）走网格**。这条写进 CI 的 `contract-check`。

## 4. Nyquist 与 G=0

* `G = 0` 的 Hartree 分量置零 ⇔ 加均匀背景电荷；中性体系精确，带电体系 V0 不支持。
* Nyquist 分量（`G = −N/2`）只计一次；`miller` 生成时用 `mod` 映射到 `[0, N)`，
  由 `build_miller` 的球判据保证不重复。

## 5. dtype 策略

见 `runtime/dtypes.py`：波函数 complex64 可接受，密度/势/能量必须 fp64，
赝势投影 fp64。任何降精度必须写进 `AccuracyLedger`，性能报告强制读取。

## 失败模式

* **F-D1**：`w_k` 用 `1/nk` 而不是归一化到 1 → 能量差 `nk` 倍，但**力仍然对**（最难发现）。
* **F-D2**：网格用 7 的光滑数（以为与 ABACUS 一致）→ 与 ABACUS `nfft` 不同，
  XC 积分误差差 0.1–1 meV/atom，且随 ecut 不收敛。
* **F-D3**：`fftn` 作用在扁平数组上（沿错轴）→ 势矩阵是垃圾，SCF 仍可能"收敛"。
  本仓库实测踩中过；检测手段：`⟨V⟩` 与 `∫v|ψ|²` 必须相等到 1e-12。

## 6. k 点约定

`KMesh` 内部存分数坐标 `kfrac`；`cartesian(lattice)=kfrac @ (2π inv(lattice).T)`。
`Σw=1`，自旋因子不进 w。V0 的 Γ 计算不受影响；非 Γ 使用联合壳层
`|k+G|²≤2Ecut` 对全部 k 取并集，并与 ABACUS 的 per-k 壳层在报告中显式区分。
