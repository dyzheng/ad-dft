"""L1 赝势 (spec/pseudopotential.md).

V0 只做**模守恒、无 NLCC** 的路径，且把「赝势的径向函数必须是 (R, cell) 的可微函数」
写成硬约束：`beta_q()` 每次调用都从解析径向函数求值，**禁止**预插值到固定 q 网格后
把结果当常量传下去（ABACUS 正是这么做的：vnl_pw_table.cpp，dq=0.010 均匀表 + 4 点插值）。
预插值会把结构因子与 |k+G| 两条导数链剪断：力会少一项，应力会丢整个「投影随基组缩放」项。
这不是精度问题，是功能缺失。

`SyntheticPseudo` 是给架构/验证跑通用的解析赝势（高斯型），
它满足 KB 形式 V_NL = Σ|β_i>D_ij<β_j|，因此非局域投影、伴随恒等式、
力/应力链路的测试都是真实的；只有「数值来自真实 UPF」这一点被推迟到 K05。
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class Pseudo(NamedTuple):
    """多物种赝势。数组的 physical 部分对微分透明（常数），几何依赖走 projectors()。"""

    zval: jnp.ndarray  # (nspec,) 价电子数
    vloc_amp: jnp.ndarray  # (nspec,) 局域势幅度
    vloc_sigma: jnp.ndarray  # (nspec,) 局域势宽度（Bohr）
    proj_l: jnp.ndarray  # (nspec, nproj) int32 —— 角动量
    proj_w: jnp.ndarray  # (nspec, nproj) 投影幅度
    proj_b: jnp.ndarray  # (nspec, nproj) 投影径向宽度（Bohr）
    dion: jnp.ndarray  # (nspec, nproj, nproj)

    @property
    def nspec(self) -> int:
        return int(self.zval.shape[0])

    @property
    def nproj(self) -> int:
        return int(self.proj_l.shape[1])


def vloc_q(psp: Pseudo, spec_idx, q):
    """局域赝势的倒空间径向部分 ṽ_loc(q)（无 1/Ω，单位 Ha）。

    spec_idx: (nat,) → 返回 (nat, nG)；spec_idx: 标量 → 返回 (nG,)。
    """
    si = jnp.atleast_1d(spec_idx)
    amp = psp.vloc_amp[si][:, None]
    sig = psp.vloc_sigma[si][:, None]
    out = -amp * jnp.exp(-0.5 * (q[None, :] * sig) ** 2)
    return out if jnp.ndim(spec_idx) > 0 else out[0]


def beta_q(psp: Pseudo, spec_idx, q):
    """非局域投影的径向部分 β_i(q)，形状 (nproj, nG)。

    **每次几何变化都重新求值**，因此 ∂β/∂q 与 ∂q/∂cell 自动进入图。
    """
    l = psp.proj_l[spec_idx].astype(q.dtype)[:, None]  # (nproj, 1)
    w = psp.proj_w[spec_idx][:, None]
    b = psp.proj_b[spec_idx][:, None]
    qq = jnp.maximum(q[None, :], 1e-12)
    return w * jnp.power(qq, l) * jnp.exp(-0.5 * (qq * b) ** 2)


def synthetic(zval=(2.0,), vloc_amp=4.0, vloc_sigma=0.8, proj_w=(1.0, 0.6),
              proj_b=(0.9, 1.4), dion_diag=(-2.0, -0.7)) -> Pseudo:
    """构造一个可解析的 KB 型赝势：l = 0, 1 两个投影子，对角 D。"""
    nspec = len(zval)
    nproj = len(proj_w)
    l = jnp.arange(nproj, dtype=jnp.int32)
    return Pseudo(
        zval=jnp.asarray(zval, dtype=jnp.float64),
        vloc_amp=jnp.full((nspec,), float(vloc_amp)),
        vloc_sigma=jnp.full((nspec,), float(vloc_sigma)),
        proj_l=jnp.tile(l[None, :], (nspec, 1)),
        proj_w=jnp.tile(jnp.asarray(proj_w), (nspec, 1)),
        proj_b=jnp.tile(jnp.asarray(proj_b), (nspec, 1)),
        dion=jnp.tile(jnp.diag(jnp.asarray(dion_diag, dtype=jnp.float64))[None, :, :], (nspec, 1, 1)),
    )


def from_upf(path: str):  # pragma: no cover - 任务卡 K05
    raise NotImplementedError(
        "K05: UPF v1/v2 解析（beta/D_ij/NLCC/索引 0-based、l^2+m 展平）尚未实现。"
        "验收：与 ABACUS pseudo.h 的 betar/dion/rho_atc 逐元素 < 1e-10。"
    )
