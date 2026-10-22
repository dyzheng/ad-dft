"""L1 赝势 (spec/pseudopotential.md).

`Pseudo` 同时承载合成赝势（`kind=0`）与真实 UPF（`kind=1`）。两条路径的字段
互不复用：合成路径只读解析参数，UPF 路径只读 `r/rab/vloc_at/betar/dion`。
唯一允许的 Ry→Ha 转换发生在 `from_upf`；`src/` 里不出现 `*_to_ry`。
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

from .upf import read_upf


class Pseudo(NamedTuple):
    """多物种赝势。数组的 physical 部分对微分透明（常数），几何依赖走 projectors()。"""

    kind: int  # 0=synthetic, 1=UPF
    zval: jnp.ndarray  # (nspec,) 价电子数
    vloc_amp: jnp.ndarray  # (nspec,) synthetic-only
    vloc_sigma: jnp.ndarray  # (nspec,) synthetic-only
    proj_w: jnp.ndarray  # (nspec,nrad) synthetic-only
    proj_b: jnp.ndarray  # (nspec,nrad) synthetic-only
    proj_l: jnp.ndarray  # (nspec,nrad) int32 —— 角动量
    dion: jnp.ndarray  # (nspec,nrad,nrad) Ha
    r: jnp.ndarray  # (nspec,mesh) 或 (nspec,1)
    rab: jnp.ndarray  # (nspec,mesh) 或 (nspec,1)
    vloc_at: jnp.ndarray  # (nspec,mesh) Ha；synthetic 填 0
    betar: jnp.ndarray  # (nspec,nrad,mesh)；synthetic 填 0
    rho_atc: jnp.ndarray  # (nspec,mesh)；synthetic 填 0
    nlcc: jnp.ndarray  # (nspec,) bool
    lmax: jnp.ndarray  # (nspec,) int32

    @property
    def nspec(self) -> int:
        return int(self.zval.shape[0])

    @property
    def nproj(self) -> int:
        return int(self.proj_l.shape[1])

    @property
    def nrad(self) -> int:
        return int(self.proj_l.shape[1])

    @property
    def nh(self):
        return jnp.asarray([
            sum(2 * int(l) + 1 for l in row) for row in self.proj_l
        ], dtype=jnp.int32)


def lm_table(proj_l_row):
    """返回 [(radial_index, l, m_index_local), ...]，m 顺序与 ABACUS 一致。"""
    tab = []
    for ib, l in enumerate(proj_l_row):
        for m in range(2 * int(l) + 1):
            tab.append((ib, int(l), m))
    return tab


def _simpson_weights(n):
    w = jnp.ones((n,), dtype=jnp.float64)
    w = w.at[1:n - 1:2].set(4.0)
    w = w.at[2:n - 1:2].set(2.0)
    return w / 3.0


def _sph_j(l, x):
    """j_l(x)，l≤2；x 可为 0。"""
    xs = jnp.where(jnp.abs(x) > 1e-10, x, 1.0)
    if l == 0:
        y = jnp.sin(xs) / xs
        return jnp.where(jnp.abs(x) > 1e-10, y, 1.0)
    if l == 1:
        y = jnp.sin(xs) / xs ** 2 - jnp.cos(xs) / xs
        return jnp.where(jnp.abs(x) > 1e-10, y, 0.0)
    y = (3.0 / xs ** 3 - 1.0 / xs) * jnp.sin(xs) - 3.0 * jnp.cos(xs) / xs ** 2
    return jnp.where(jnp.abs(x) > 1e-10, y, 0.0)


def vloc_q(psp: Pseudo, spec_idx, q):
    """局域赝势的倒空间径向部分 ṽ_loc(q)（无 1/Ω，单位 Ha）。

    spec_idx: (nat,) → 返回 (nat, nG)；spec_idx: 标量 → 返回 (nG,)。
    """
    si = jnp.atleast_1d(jnp.asarray(spec_idx))
    if psp.kind == 0:
        amp = psp.vloc_amp[si][:, None]
        sig = psp.vloc_sigma[si][:, None]
        out = -amp * jnp.exp(-0.5 * (q[None, :] * sig) ** 2)
    else:
        rows = [_vloc_q_upf(psp, int(s), q) for s in np.asarray(si)]
        out = jnp.stack(rows, axis=0)
    return out if jnp.ndim(spec_idx) > 0 else out[0]


def _vloc_q_upf(psp: Pseudo, si: int, q):
    r, rab = psp.r[si], psp.rab[si]
    v = psp.vloc_at[si] / 2.0  # Ry -> Ha
    z = psp.zval[si]
    w = _simpson_weights(r.shape[0])
    qs = jnp.where(q > 1e-12, q, 1.0)
    aux = r * v + z * erf(r)
    integ = jnp.sum(w[None, :] * rab[None, :] * aux[None, :] *
                    jnp.sin(qs[:, None] * r[None, :]) / qs[:, None], axis=1)
    integ = integ - z * jnp.exp(-qs ** 2 / 4.0) / qs ** 2
    g0 = jnp.sum(w * rab * r * (r * v + z))
    integ = jnp.where(q > 1e-12, integ, g0)
    return 4.0 * jnp.pi * integ


def beta_q(psp: Pseudo, spec_idx, q):
    """非局域投影的径向部分 β_i(q)，形状 (nrad, nG)。

    **每次几何变化都重新求值**，因此 ∂β/∂q 与 ∂q/∂cell 自动进入图。
    """
    if psp.kind == 0:  # 合成赝势：保留解析路径
        si = int(spec_idx)
        l = psp.proj_l[si].astype(q.dtype)[:, None]
        w = psp.proj_w[si][:, None]
        b = psp.proj_b[si][:, None]
        qq = jnp.maximum(q[None, :], 1e-12)
        return w * jnp.power(qq, l) * jnp.exp(-0.5 * (qq * b) ** 2)
    si = int(spec_idx)
    r, rab, betar = psp.r[si], psp.rab[si], psp.betar[si]
    w = _simpson_weights(r.shape[0])
    out = []
    for ib in range(psp.nrad):
        jl = _sph_j(int(psp.proj_l[si, ib]), q[:, None] * r[None, :])
        integral = jnp.sum(w[None, :] * rab[None, :] *
                           betar[ib][None, :] * r[None, :] * jl, axis=1)
        out.append(4.0 * jnp.pi * integral)  # 与 ABACUS tab 的前因子对齐
    return jnp.stack(out, axis=0)


def synthetic(zval=(2.0,), vloc_amp=4.0, vloc_sigma=0.8, proj_w=(1.0, 0.6),
              proj_b=(0.9, 1.4), dion_diag=(-2.0, -0.7)) -> Pseudo:
    """构造一个可解析的 KB 型赝势：l = 0, 1 两个投影子，对角 D。"""
    nspec = len(zval)
    nproj = len(proj_w)
    l = jnp.arange(nproj, dtype=jnp.int32)
    zeros = jnp.zeros((nspec, 1), dtype=jnp.float64)
    return Pseudo(
        kind=0,
        zval=jnp.asarray(zval, dtype=jnp.float64),
        vloc_amp=jnp.full((nspec,), float(vloc_amp)),
        vloc_sigma=jnp.full((nspec,), float(vloc_sigma)),
        proj_l=jnp.tile(l[None, :], (nspec, 1)),
        proj_w=jnp.tile(jnp.asarray(proj_w), (nspec, 1)),
        proj_b=jnp.tile(jnp.asarray(proj_b), (nspec, 1)),
        dion=jnp.tile(jnp.diag(jnp.asarray(dion_diag, dtype=jnp.float64))[None, :, :], (nspec, 1, 1)),
        r=zeros,
        rab=zeros,
        vloc_at=zeros,
        betar=jnp.zeros((nspec, nproj, 1), dtype=jnp.float64),
        rho_atc=zeros,
        nlcc=jnp.zeros((nspec,), dtype=bool),
        lmax=jnp.full((nspec,), int(l.max()), dtype=jnp.int32),
    )


def from_upf(path: str) -> Pseudo:
    raw = read_upf(path)
    if raw["nlcc"]:
        raise NotImplementedError(
            "NLCC 文件暂不接受（ABACUS 自身也未在 XC 中使用 rho_atc）。"
            "本轮只允许 nlcc=False 的 UPF；见 spec/pseudopotential.md。")
    betar = jnp.asarray(raw["betar"])
    nrad = betar.shape[0]
    return Pseudo(
        kind=1, zval=jnp.asarray([raw["zval"]]),
        vloc_amp=jnp.zeros((1,), dtype=jnp.float64),
        vloc_sigma=jnp.ones((1,), dtype=jnp.float64),
        proj_w=jnp.zeros((1, nrad), dtype=jnp.float64),
        proj_b=jnp.ones((1, nrad), dtype=jnp.float64),
        proj_l=jnp.asarray(raw["lll"])[None, :],
        dion=jnp.asarray(raw["dion"])[None, :, :] / 2.0,  # Ry -> Ha
        r=jnp.asarray(raw["r"])[None, :], rab=jnp.asarray(raw["rab"])[None, :],
        vloc_at=jnp.asarray(raw["vloc_at"])[None, :],
        betar=betar[None, :, :], rho_atc=jnp.asarray(raw["rho_atc"])[None, :],
        nlcc=jnp.asarray([raw["nlcc"]]),
        lmax=jnp.asarray([max(raw["lll"])], dtype=jnp.int32),
    )
