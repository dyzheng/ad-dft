# E3: Direct minimization of the variational energy

## 假设

不做 SCF、不做 eigh，直接对轨道 `C` 最小化能量，使用 matrix-free `Hψ`。

## 原型

- 能量：`E_var = Σf<C|Hψ> − E_H + E_xc − <C|V_xc|C> + E_II`
- 固定占据（Si 4 条占据带）
- Riemannian 投影梯度 + 回溯线搜索
- JAX `value_and_grad`
- 初值：SCF 解 + 1e-4 随机扰动

## 结果（20 Ry，`nG=411`）

| 量 | 值 |
|---|---:|
| SCF 能量 | -7.2846117406 Ha |
| 初值能量误差 | 3.65e-4 Ha |
| 3 步后 | 1.19e-4 Ha |
| 之后 | 线搜索停滞 |

从 1e-2 扰动出发更差。简单梯度下降/线搜索不足以到 1e-8 Ha。

## 结论

- 直接最小化的梯度方向/能量公式已验证（`-conj(g)` 方向下降）；
- 但条件数差，必须用 **Riemannian L-BFGS / trust-region + 预条件**；
- 当前原型不能替换 SCF。

## 决策

**Yellow**。下一步：

1. 预条件 `(T + α)^{-1}`；
2. Riemannian L-BFGS / trust-region；
3. 与 Davidson/LOBPCG 对比；失败则回退 SCF。
