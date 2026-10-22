# 契约 1：单位与常数

> 本文件的任何改动都要求重跑全部 5 级对齐（`docs/VALIDATION.md`）。
> 单位错误是唯一一种"能量看起来对、梯度全错"的来源。

## 内部单位制

| 量 | 单位 | 数值约定 |
|---|---|---|
| 长度 | Bohr | `1 Bohr = 0.529177210903 Å` |
| 能量 | Hartree | `1 Ha = 27.211386245988 eV = 2 Ry` |
| 电荷 | e | `e² = 1`（原子单位） |
| 质量 | m_e | `m_e = 1` |
| ħ | 1 | 动能算符 `T = -½∇²` |
| 应力 | Ha/Bohr³ | `1 Ha/Bohr³ = 14710.516 kbar` |

**ABACUS 内部是 Ry–Bohr 制**（`source_base/constants.h:79` `e2 = 2.0`；
`op_pw_ekin.cpp:45` 动能算符 `gk2*tpiba2` 无 ½ 因子）。因此：

```
E_Ha = E_Ry / 2
ecut_Ha = ecutwfc_Ry / 2
sigma_Ha = smearing_sigma_Ry / 2
```

转换**只允许**出现在 `adapters/abacus/` 与输出格式化处。CI 的 `contract-check`
扫描 `src/` 下是否出现 `*_to_ry` / `*_to_ha` 的调用，出现即失败。

## UPF / libxc 边界

- UPF `PP_LOCAL`、`PP_DIJ`、`betar` 原始为 Ry 制；`from_upf` 只在这里做 `/2`。
- libxc 使用 Ha（`xc_lda_exc_vxc` 的 `zk` 是 Ha/粒子）；不引入第二套单位。
- `beta_q` 含 ABACUS 表的 `4π` 前因子；`local_potential_g` 再除 `Ω`。

## 倒空间约定

```
a_i · b_j = 2π δ_ij          （reciprocal() 的行向量）
ψ_nk(r) = (1/√Ω) Σ_G c_nk(G) e^{i(k+G)·r}
n(r)    = Σ_nk w_k f_nk g_s |ψ_nk(r)|²
n(G)    = (1/Ω) ∫ n(r) e^{-iG·r} dr = fftn(n)/N
V_H(G)  = 4π n(G)/G²  (G≠0),  V_H(0) = 0
V_NL    = Σ_I Σ_ij |β_Ii⟩ D_Iij ⟨β_Ij|,
          β_Ii(G) = (1/√Ω) β̃_i(|k+G|) e^{-i(k+G)·R_I}
```

三条必须锁死的约定（任何一条换了，能量不变、力会错）：

1. `1/√Ω` 的指数是 **−1/2**（应力体积项系数 = −1/2，不是 −1）；
2. 结构因子符号 **e^{−iG·R}**，与局域势、非局域投影、Ewald 一致；
3. FFT 归一化：正变换带 `1/N`，逆变换不带（`jnp.fft` 语义），
   代码里通过 `N/√Ω` 与 `√Ω/N` 两个因子吸收。

## 泛函导数的离散权重

离散积分近似 `∫ f dr ≈ (Ω/N) Σ_j f_j`。因此

```
δE/δn_j = (N/Ω) · ∂E/∂n_j        ← 泛函导数 ≠ 偏导数
```

`jax.grad(E_xc)` 给出的是 `(Ω/N)·f'(n_j)`；物理势是 `f'(n_j)`。
漏掉 `N/Ω` 会让 XC 势小 8 倍（N=1728, Ω=216 时），而**总能仍然"看起来合理"**。
本仓库实测：该错误使力偏差 25%，且只有"能量对密度的驻点性"测试能抓到。

## 数值容差（写死，不允许在调用点改）

| 量 | fp64 目标 | fp32 目标 |
|---|---|---|
| 原语逐元素 | < 1e-10 | < 1e-5 |
| 伴随恒等式 | < 1e-14（机器精度） | — |
| 能量 vs 参考 | < 1 meV/atom | < 5 meV/atom |
| 力 vs 中心差分 | < 1e-4 eV/Å | — |
| 应力 vs 应变差分 | < 0.1 kbar | — |
| SCF 密度残差 | < 1e-10（归一化） | < 1e-7 |

## 失败模式

* **F-U1**：`ecutwfc` 未除 2 直接当 Ha 用 → 基组大一倍，能量系统性偏低。
  检测：`shell_margin()` 与 `nG` 和 ABACUS 对比。
* **F-U2**：`σ` 用 eV 当 Ha 用 → 金属占据数错 27 倍，力偏差几十 meV/Å。
* **F-U3**：泛函导数漏 `N/Ω` → 见上，只有驻点性测试能抓。
