# 从 ABACUS 源码逐条确认的事实

> 目的：把"方案文档的假设"换成"源码的事实"。每条都带文件与行号，
> 便于复核。仓库：`abacus-develop`（develop 分支）。

## 1. 单位：内部是 Ry–Bohr，不是 Ha

* `source/source_base/constants.h:79` — `const double e2 = 2.0;`（Ry 制下 e²=2）
* `source/source_pw/module_pwdft/op_pw_ekin.cpp:45` — 动能算符 `tmhpsi += gk2*tpiba2*psi`，**无 ½**
* `source/source_io/input_help.cpp:462` — `ecutwfc` 单位 Ry
* 全树无 `Ry_to_Ha` 转换；`Ry_to_eV = 13.605698` 仅用于输出

⇒ 本项目内部用 Ha，桥必须在 adapter 层，且 `E_Ha = E_Ry/2`。

## 2. FFT 盒规则：奇数 → 只含 2,3,5

* `source/source_basis/module_pw/pw_init.cpp:23` `PW_Basis::initgrids`
* `:92-137` 自增直到 `b==1`，其中 7 的分支被注释：`//if (b%7==0){ n7++; ... }`
* `:110-113` `full_pw_dim` 可强制奇/偶
* 密度网格 `ecutrho = 4×ecutwfc`：`source/source_io/module_parameter/read_input_item_system.cpp:915`
* 波函数网格复用密度网格：`source/source_pw/module_pwdft/setup_pwwfc.cpp:60-66`

## 3. 赝势

* 解析：`source/source_cell/read_pp.h:25` `Pseudopot_upf`；`read_pp_upf100.cpp`（v1）、`read_pp_upf201.cpp`（v2）
* 存储：`source/source_cell/pseudo.h:58-84` — `rho_atc`、`vloc_at`、`lll`、`jjj`、`betar(nbeta,mesh)`、`dion`、`nh=Σ(2l+1)`
* 索引 **0-based**：`read_pp_upf201.cpp:474,527,545`
* 例外：`vnl_pw.cpp:523` 的 `ijtoh` 残留 1-based，源码注释 "I'm not sure from 0 or 1"
* **m 通道展开**：每个径向 β 按 `m=0..2l` 展开为 `2l+1` 个投影子；每 species 的
  `nh = Σ(2l+1)`，D 对同一 `(l,m)` 复制、不同 `(l,m)` 之间为零。
  `vnl_pw.cpp:494-506` 用 `nhtolm(it,ih) = l*l + m` 展平；Si 的
  `l=0,1` 两个径向投影子对应 `nh=1+3=4`（每原子），而非 2。
* **k 点单位**：`K_Vectors.kvec_d` 是分数坐标，`kvec_c = kvec_d · reciprocal_vec`
  （`source/source_cell/klist.cpp:93`）；平面波动能始终用笛卡尔 `|k+G|²`
  （`tpiba2`），不能把 `2π·frac` 直接当 k。
* **β 预表化**：`vnl_pw.cpp:721-757` 填 `tab`，`dq=0.010`（`system_parameter.h:29`），
  `nqx = int((sqrt(ecutwfc)/dq + 4.0)*cell_factor)`（`read_set_globalv.cpp:54`）；
  取值插值在 `kernels/vnl_op.cpp:45-60`（`_polynomial_interpolation`）
* **NLCC 未参与 XC**：`rho_atc` 在 `read_pp_upf201.cpp:30-33` 读入、`atom_pseudo.cpp:208` 广播、
  `pseudo.cpp:62` 仅打印；全树无消费点

## 4. 哈密顿量与求解

* 组装顺序：`source/source_pw/module_pwdft/hamilt_pw.cpp:45-200`
  `Ekinetic → Veff → (Meta) → Nonlocal → (OnsiteProj) → (EXX)`
* 局域+Hartree+XC 合成一个实空间乘子：`source/source_pw/module_pwdft/op_pw_veff.cpp`
* V_NL 施加：`source/source_pw/module_pwdft/op_pw_nl.cpp:59-127` 调 `nonlocal_op()`
  （`kernels/vnl_op.cpp`），三步批处理 GEMM（`becp` → 乘 `D` → 回乘 `vkb`），
  一次覆盖全部能带×投影子
* PW 对角化器：`source/source_hsolver/hsolver_pw.cpp:90` `{"cg","dav","dav_subspace","bpcg"}`

## 5. SCF 与占据

* 主循环：`source/source_esolver/esolver_ks.cpp:187-230`
* 混合器：`source/source_estate/module_charge/charge_mixing.cpp:115-128` `broyden / plain / pulay`
  + Kerker 预条件 `charge_mixing_preconditioner.cpp:6,77`
* **收敛判据**：`chgmixing.cpp:41`
  `conv_esolver = (drho < scf_thr && not_restart_step && converged_u);`
  `drho` 定义在 `charge_mixing_residual.cpp:7-71`：`scf_thr_type==2` 时是归一化密度残差
* 占据：`source/source_estate/occupy.cpp:95-132` `gauss/mp/mv/cold/fd`；
  **μ 用纯二分**（`:394-464`，maxiter=300）；`sigma` 单位 **Ry**，默认 0.015

## 6. 能量、力、应力

* 总能求和：`source/source_estate/fp_energy.cpp:13`
  `etot = eband + deband + (etxc - etxcc) + ewald + hartree + demet + descf + exx + efield`
* 力的 5 项：`source/source_pw/module_pwdft/forces.cpp:61-93`
  `cal_force_loc → cal_force_nl → cal_force_onsite → cal_force_cc → cal_force_scc`
* 应力 10 项：`stress_pw.cpp:86-126`
  `kin/har/ewa/gga/mgga/loc/cc/nl/us/exx`

## 7. k 点与对称性

* `K_Vectors`：`source/source_cell/klist.h:11`；IBZ 约化 `k_vector_utils.cpp:353`
* **不用 spglib**：自研 `source/source_cell/module_symmetry/symm_getgroup.cpp`（最多 48 个操作）

## 8. 对本项目的三条直接后果

1. **应力对比要小心**：ABACUS 的 β 预表化切断了 `∂β/∂cell`，我们重建则不切断。
   某些体系上"我们的应力 ≠ 他们的应力"不是 bug，而是我们多了一项。
2. **NLCC 必须显式决策**：ABACUS 读了不用。我们要么实现、要么显式拒绝，
   不允许静默跟随——否则差异预算表里那一行永远归因不了。
3. **单位桥只允许一处**：`ecutwfc`、`sigma` 都是 Ry；内部全 Ha。
   任何在 `src/` 里出现的 `*_to_ry` 都是缺陷（CI 检查）。
