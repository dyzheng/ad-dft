# 契约 3：赝势（UPF）与投影

## 1. 与 ABACUS 的对齐点（从源码确认）

| 项目 | ABACUS 事实 | 来源 |
|---|---|---|
| 解析类 | `Pseudopot_upf`（v1: upf100 / v2: upf201） | `source_cell/read_pp*.cpp` |
| 存储 | `rho_atc`、`vloc_at`、`r`、`rab`、`lll`、`jjj`、`betar(nbeta,mesh)`、`dion(nbeta,nbeta)`、`nh=Σ(2l+1)` | `pseudo.h:58-84` |
| 索引 | **0-based**（`ib=0..nbeta-1`） | `read_pp_upf201.cpp:474` |
| l,m 展平 | `nhtolm(it,ih) = l*l + m`，`ib` 外层、`m=0..2l` 内层 | `vnl_pw_init_vnl.cpp:86-95` |
| 例外 | `ijtoh` 残留 1-based，源码自带注释 "I'm not sure from 0 or 1" | `vnl_pw_init_vnl.cpp:113` |
| β(|k+G|) | **预表化**：均匀 q 网格 `dq=0.010`，`nqx=int((sqrt(ecut)/dq + 4)*cell_factor)`，4 点插值 | `vnl_pw_table.cpp:11`、`vnl_pw_getvnl.cpp:88` |
| NLCC | `rho_atc` 被读取、广播、**仅打印**；全树无消费点 → XC 中未使用 | `read_pp_upf201.cpp:30`、`pseudo.cpp:66` |

## 2. 本项目的差异（必须显式声明）

1. **不预表化**。β̃_i(q) 每次几何变化从解析径向函数求值（V0 重建，G2 换可微插值）。
   原因：预插值切断 `∂β/∂q · ∂q/∂cell` 两条链，力少一项、应力丢整个缩放项。
   **验收**：同一赝势下，预表化路径与本路径的能量差 < 0.1 meV/atom，
   且本路径在应变下的应力与有限差分一致（预表化路径不会一致）。
2. **NLCC 要么实现、要么显式拒绝**。不允许"读了但不用"这种静默行为。
   V0 选择：解析 `rho_atc` 并在 XC 密度上叠加；与 ABACUS 对比时把该差异
   列入差异预算（1–5 meV/atom），并在报告头显式标记 `nlcc=on/off`。

## 3. 接口

```python
class Pseudo(NamedTuple):     # 静态数据，不参与微分
    zval, vloc_amp, vloc_sigma,
    proj_l, proj_w, proj_b, dion

vloc_q(psp, spec_idx, q)      # (nat, nG) 或 (nG,)
beta_q(psp, spec_idx, q)      # (nproj, nG) —— 每次重建
projectors(psp, spec_idx, basis, cell, kpts) -> (nk, nat*nproj, nG)
```

`D` 的块对角构造必须是 `(nat, nat, nproj, nproj)` 再 reshape，
**不能**用 `(nat, nproj, nproj).reshape(nat*npj, nat*npj)`（会把块拼错）。

## 4. 失败模式

* **F-P1**：`D_ij` 索引与 `β` 的 `(l,m)` 展平顺序不匹配 → 能量差几十 meV，力差一个量级。
* **F-P2**：结构因子符号反（`e^{+iG·R}`）→ 力符号正确、大小错（因为 |β|² 对称）。
* **F-P3**：`1/√Ω` 写成 `1/Ω` → 能量随体积线性漂移，应力错一个量级。
* **F-P4**：预插值后把 β 当常量 → 力丢 Pulay 项，应力丢缩放项。
  检测：`⟨Ax,y⟩ = ⟨x,A^T y⟩` 检查不出这个（β 被当常量后算子仍 Hermitian）；
  只有"对 R 的 VJP vs 中心差分 × 三个构型"能抓。
