"""L2 原语 · Hartree (spec/discretization.md §g0).

V_H(G) = 4π n(G)/G² (G≠0),  V_H(0) = 0.
G=0 置零等价于「加均匀背景电荷」，对中性体系精确；对带电体系需要额外修正（V0 不支持）。

线性算子：VJP 即其伴随。因此这里不做任何手工求导——
`hartree_energy` 与 `hartree_potential_r` 是同一个离散泛函的两种取值方式，
测试要求两者在机器精度内一致（自洽性 oracle）。
"""

from __future__ import annotations

import jax.numpy as jnp

from ..geometry.basis import Basis, density_g, g2, real_from_g
from ..geometry.cell import volume

FOUR_PI = 4.0 * jnp.pi


def _vh_g(basis: Basis, lattice, rho_r):
    ng = density_g(basis, lattice, rho_r)
    q2 = g2(basis, lattice)
    safe = jnp.where(q2 > 1e-14, q2, 1.0)
    vg = jnp.where(q2 > 1e-14, FOUR_PI * ng / safe, 0.0)
    return vg, ng, q2


def hartree_energy(basis: Basis, lattice, rho_r):
    vg, ng, q2 = _vh_g(basis, lattice, rho_r)
    omega = volume(lattice)
    return 0.5 * omega * jnp.sum(jnp.where(q2 > 1e-14, jnp.real(ng * jnp.conj(vg)), 0.0))


def hartree_potential_r(basis: Basis, lattice, rho_r):
    """实空间 Hartree 势：v(r) = Σ_G V_G e^{iG·r} = N · ifftn(V_grid)。

    **V_G 必须以 complex 进入 FFT。** 这里原本有一句 `vg.astype(rho_r.dtype)`，
    它把 complex128 静默转成 float64，等于丢掉 n_G 的虚部——而实密度满足
    n(-G) = n(G)*，虚部一般不为零。丢掉之后 Hermitian 对称性被破坏，
    得到的势等价于"密度先被对称化、再求势"：SCF 仍然收敛、总能只差几 mHa，
    但整条梯度链（力、应力、响应算子）全错，而且能量形式之间的恒等式会悄悄失效。

    这一类错误只有"v(ρ+δ) − v(ρ) 是否严格等于 v(δ)"能抓到——见
    tests/test_primitives.py::test_hartree_exact_linearity。
    """
    vg, _, _ = _vh_g(basis, lattice, rho_r)
    return real_from_g(basis, lattice, vg)


def hartree_gaussian_check(alpha: float):
    """解析 oracle：ρ(r) = (α/π)^{3/2} e^{-α r²} → V_H(r) 的解析值。

    返回 v_exact(r)，供 tests/test_hartree.py 使用（G 空间截断误差随 ecut 收敛）。
    """
    import math

    def v_exact(r):
        return math.erf(math.sqrt(alpha) * r) / r if r > 0 else 2.0 * math.sqrt(alpha / math.pi)

    return v_exact
