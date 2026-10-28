"""L3 求解器 · 哈密顿量组装、SCF 不动点与混合器。

分层的关键约定：
  * 混合器 (mixer) 是**加速器不是物理**，永远不进微分图；
  * SCF 的「物理映射」`scf_map` 才是反向传播的对象：n ↦ 对角化 ↦ 占据 ↦ 密度；
  * 收敛判据用**归一化密度残差**（与 ABACUS chg_drho.cpp 同口径），不用能量差。
    能量差小但密度残差大时，梯度不可信。

目标函数（显式泛函，不含 double counting）：
    E = T[ψ] + ∫v_loc n + E_H[n] + E_xc[n] + E_NL[ψ] + E_II
这个形式的好处是：每个 term 都是 n 与 ψ 的显式函数，力的三项 Pulay 全由链式法则给出。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..geometry.basis import Basis, g2, gvectors, psi_to_real
from ..geometry.cell import Cell, volume
from ..physics import hartree, xc
from ..physics.density import density_from_psi
from ..physics.local import local_potential_r
from ..physics.occupation import occupations
from ..system import SCFState, Static
from .eigh import eigh_lowest
from .geometry_cache import build_geometry_cache


# ------------------------------------------------------------------ 哈密顿量

def vmat_from_real(basis: Basis, v_r):
    """<G'|V|G> = fft(V)[(m_G − m_G') mod N] / N。实空间势 → 稠密矩阵，无额外约定。"""
    n1, n2, n3 = basis.fft_shape
    vg = jnp.fft.fftn(v_r.reshape(n1, n2, n3)) / basis.nfft
    m = basis.miller
    d = m[:, None, :] - m[None, :, :]
    i1 = jnp.mod(d[..., 0], n1)
    i2 = jnp.mod(d[..., 1], n2)
    i3 = jnp.mod(d[..., 2], n3)
    idx = (i1 * n2 + i2) * n3 + i3
    return vg.reshape(-1)[idx]


def kinetic_diag(basis: Basis, lattice, kpt):
    kg = gvectors(basis, lattice, kpt)
    return 0.5 * jnp.sum(kg * kg, axis=-1)


def hamiltonian(geo: Cell, static: Static, v_r, cache=None):
    """返回 (nk, nG, nG) 的 H = T + V_loc + V_H + V_xc + V_NL。

    V_NL 必须出现在 H 里——否则本征值缺非局域项，而能量分解里又有它，
    两者不自洽（实测差 0.046 Ha，正好等于 E_NL）。这是「能量对、梯度错」的
    另一种形态：两种能量写法不一致。

    ``cache`` 只用于正向 SCF 的几何复用；JVP 路径必须传 ``None``。
    """
    from ..physics import nonlocal_op

    basis = static.basis
    vm = vmat_from_real(basis, v_r)
    if cache is None:
        proj = nonlocal_op.projectors(static.psp, static.spec_idx, basis, geo, static.kpts)
        d = nonlocal_op.dion_full(static.psp, static.spec_idx)
    else:
        proj, d = cache.proj, cache.dion
    hs = []
    for ik in range(int(static.kpts.shape[0])):
        t = kinetic_diag(basis, geo.lattice, static.kpts[ik])
        vnl = jnp.einsum("pg,pq,qh->gh", proj[ik], d, jnp.conj(proj[ik]))
        h = jnp.diag(t) + vm + vnl
        hs.append(0.5 * (h + jnp.conj(h.T)))  # 显式 Hermitian 化（数值）
    return jnp.stack(hs, axis=0)


def effective_potential(geo: Cell, static: Static, rho_r, with_xc=True, cache=None):
    basis = static.basis
    if cache is None:
        v_loc = local_potential_r(static.psp, static.spec_idx, basis, geo)
    else:
        v_loc = cache.v_loc_r
    v_h = hartree.hartree_potential_r(basis, geo.lattice, rho_r)
    if with_xc:
        v_xc = xc.xc_potential_r(xc.xc_energy_lda, basis, geo.lattice, rho_r)
    else:
        v_xc = jnp.zeros_like(v_h)
    return v_loc + v_h + v_xc, (v_loc, v_h, v_xc)


# ------------------------------------------------------------------ 物理映射

def _hamiltonian_from_density(geo: Cell, static: Static, rho_r, cache=None):
    v_r, _ = effective_potential(geo, static, rho_r, cache=cache)
    return hamiltonian(geo, static, v_r, cache=cache)


def _hamiltonian_and_tangent(geo: Cell, static: Static, rho_r, dgeo, drho):
    """H[n] 与其方向导数，避开 libxc custom_vjp 的 forward-mode 限制。

    H 对密度的依赖只经 v_H 与 v_xc。``v_xc`` 的密度导数用 LDA 对角核，
    ``v_H`` 与几何项走平滑 JVP；``hamiltonian`` 本身在固定 ``v_r`` 下对
    geometry 求导，再补上 ``v_r`` 的密度/局域势变化。这样既不穿过 eigh，
    也不要求 ctypes XC 支持 forward-mode。
    """
    v_r, _ = effective_potential(geo, static, rho_r)
    H = hamiltonian(geo, static, v_r)
    _, dH_geo = jax.jvp(
        lambda g, v: hamiltonian(g, static, v),
        (geo, v_r), (dgeo, jnp.zeros_like(v_r)))
    _, dv_loc = jax.jvp(
        lambda g: local_potential_r(static.psp, static.spec_idx,
                                    static.basis, g),
        (geo,), (dgeo,))
    _, dv_h = jax.jvp(
        lambda lat, r: hartree.hartree_potential_r(static.basis, lat, r),
        (geo.lattice, rho_r), (dgeo.lattice, drho))
    dv_xc = xc.xc_kernel_r(rho_r) * drho
    dH = dH_geo + vmat_from_real(static.basis, dv_loc + dv_h + dv_xc)
    return H, dH


def _density_tangent_from_dh(geo: Cell, static: Static, H, dH, dlat, tol=1e-6):
    """固定输入密度时，由一阶微扰 δH 给出的 δF。

    本征矢响应按 ``δC_n = Σ_m C_m M_mn/(ε_n − ε_m)`` 计算；
    ``|ε_n − ε_m| < tol`` 的子空间内旋转不改变总密度，因此置零。
    这样不需要对 ``eigh`` 反向求导，Γ 点简并不会产生 1/0。
    """
    basis = static.basis
    lat = geo.lattice
    nband = int(static.nband)
    deg = static.spin_degeneracy
    eps, C = jnp.linalg.eigh(H)  # (nk,nG), (nk,nG,nG)；列为本征矢
    vec = jnp.swapaxes(C[..., :, :nband], -1, -2)
    eig = eps[..., :nband]
    occ, _ = occupations(eig, static.kweights, static.nelec,
                         static.sigma, deg)

    M = jnp.einsum("kmi,kmn,knj->kij", jnp.conj(C), dH, C)
    M = 0.5 * (M + jnp.conj(M).transpose(0, 2, 1))
    deps = jnp.real(jnp.diagonal(M, axis1=-2, axis2=-1))[..., :nband]
    if static.sigma > 1e-6:
        fprime = occ * (1.0 - occ) / static.sigma
        denom = jnp.sum(static.kweights[:, None] * fprime, axis=-1) + 1e-30
        dmu = jnp.sum(static.kweights[:, None] * fprime * deps, axis=-1) / denom
        docc = fprime * (deps - dmu[:, None])
    else:
        docc = jnp.zeros_like(occ)

    d = eps[:, None, :] - eps[:, :, None]  # d[m,n] = ε_n − ε_m
    off = jnp.abs(d) > tol
    wgt = jnp.where(off, 1.0 / jnp.where(off, d, 1.0), 0.0)
    coef = M * wgt
    dC = jnp.einsum("kmi,kin->kmn", C, coef[..., :nband])
    dvec = jnp.swapaxes(dC, -1, -2)

    psi_r, dpsi_r = jax.jvp(
        lambda l, c: psi_to_real(basis, l, c),
        (lat, vec), (dlat, dvec))
    dmod2 = 2.0 * jnp.real(jnp.conj(psi_r) * dpsi_r)
    w = static.kweights[:, None, None] * deg
    return jnp.sum(w * (occ[..., None] * dmod2
                        + docc[..., None] * jnp.abs(psi_r) ** 2), axis=(0, 1))


@jax.custom_jvp
def scf_map(geo: Cell, static: Static, rho_r, cache=None):
    """一次物理映射：n → H[n] → 对角化 → 占据 → n'。

    这是反向传播唯一允许看到的映射：混合器不在这里。
    ``cache`` 只允许在正向 SCF 内使用；微分路径必须传 ``None``。
    """
    H = _hamiltonian_from_density(geo, static, rho_r, cache=cache)
    eig, vec = eigh_lowest(H, static.nband)
    occ, mu = occupations(eig, static.kweights, static.nelec, static.sigma,
                          static.spin_degeneracy)
    return density_from_psi(static.basis, geo.lattice, vec, occ,
                            static.kweights, static.spin_degeneracy)


@scf_map.defjvp
def _scf_map_jvp(primals, tangents):
    geo, static, rho_r, _cache = primals
    dgeo, _, drho, _ = tangents
    # 几何缓存对旧 geo 成立，切空间里不能用；JVP 一律从当前 geo 重建。
    H, dH = _hamiltonian_and_tangent(geo, static, rho_r, dgeo, drho)
    eig, vec = eigh_lowest(H, static.nband)
    occ, _ = occupations(eig, static.kweights, static.nelec, static.sigma,
                         static.spin_degeneracy)
    rho_out = density_from_psi(static.basis, geo.lattice, vec, occ,
                               static.kweights, static.spin_degeneracy)
    return rho_out, _density_tangent_from_dh(geo, static, H, dH, dgeo.lattice)


# ------------------------------------------------------------------ 混合器

def q2_grid(basis: Basis, lattice):
    """全 FFT 网格上的 |G|^2（用于预条件子与滤波，不进微分图）。"""
    n1, n2, n3 = basis.fft_shape
    i1 = jnp.arange(n1); i2 = jnp.arange(n2); i3 = jnp.arange(n3)
    m1 = jnp.where(i1 < n1 // 2, i1, i1 - n1)
    m2 = jnp.where(i2 < n2 // 2, i2, i2 - n2)
    m3 = jnp.where(i3 < n3 // 2, i3, i3 - n3)
    mm = jnp.stack(jnp.meshgrid(m1, m2, m3, indexing="ij"), axis=-1).astype(jnp.float64)
    bmat = 2.0 * jnp.pi * jnp.linalg.inv(lattice).T
    g = mm @ bmat
    return jnp.sum(g * g, axis=-1)


def kerker_mix(basis: Basis, lattice, rho_in, rho_out, alpha=0.5, gg0=1.0):
    """Kerker 预条件 + 线性混合（G 空间）。只影响正向收敛速度，不进微分图。"""
    shape = tuple(basis.fft_shape)
    dn = (rho_out - rho_in).reshape(shape)
    dg = jnp.fft.fftn(dn) / basis.nfft
    q2 = q2_grid(basis, lattice)
    filt = jnp.where(q2 > 1e-12, q2 / (q2 + gg0), 0.0)
    dn_f = (basis.nfft * jnp.fft.ifftn(dg * filt)).real.reshape(basis.nfft)
    return rho_in + alpha * dn_f


def density_residual(basis: Basis, lattice, rho_new, rho_old, nelec):
    """ABACUS scf_thr_type=2 同口径：Σ|Δρ|·Ω/N / N_e。"""
    omega = volume(lattice)
    return jnp.sum(jnp.abs(rho_new - rho_old)) * omega / basis.nfft / nelec


# ------------------------------------------------------------------ SCF 驱动

def initial_density(static: Static, geo: Cell):
    omega = volume(geo.lattice)
    return jnp.full((static.basis.nfft,), static.nelec / omega, dtype=jnp.float64)


def _solve_scf_python(geo: Cell, static: Static, rho0=None, alpha=0.5,
                      gg0=1.0, tol=1e-10, max_iter=200, verbose=False,
                      mixer="diis", beta=0.5, cache=None):
    """Python 驱动的参考 SCF（DIIS/Kerker）。作为 device driver 的回退。"""
    from .mixer import DIISMixer

    basis = static.basis
    rho = initial_density(static, geo) if rho0 is None else rho0
    if cache is None:
        cache = build_geometry_cache(geo, static)
    mix = DIISMixer(basis.nfft, m=8, beta=beta) if mixer == "diis" else None
    res = jnp.asarray(1.0)
    prev = 1e9
    it = 0
    for it in range(1, max_iter + 1):
        rho_out = scf_map(geo, static, rho, cache)
        res = density_residual(basis, geo.lattice, rho_out, rho, static.nelec)
        if mixer == "diis":
            if float(res) > prev * 1.5:      # 振荡检测 → restart 历史
                mix.reset()
            prev = float(res)
            rho = mix(rho, rho_out)
        else:
            rho = kerker_mix(basis, geo.lattice, rho, rho_out, alpha, gg0)
        if verbose and (it % 10 == 0 or it < 3):
            print(f"  scf {it:3d}  drho={float(res):.3e}")
        if float(res) < tol:
            break
    # 收敛点"抛光"：把混合后的 ρ 直接代回物理映射，直到它是严格不动点。
    # 包络定理要求 ψ 与 ρ 严格一致；否则响应项会退化成 1e-3 量级的假信号。
    for _ in range(4):
        rho_new = scf_map(geo, static, rho, cache)
        res = density_residual(basis, geo.lattice, rho_new, rho, static.nelec)
        rho = rho_new
        if float(res) < 1e-14:
            break

    v_r, _ = effective_potential(geo, static, rho, cache=cache)
    H = hamiltonian(geo, static, v_r, cache=cache)
    eig, vec = eigh_lowest(H, static.nband)
    occ, mu = occupations(eig, static.kweights, static.nelec, static.sigma,
                          static.spin_degeneracy)
    return SCFState(psi=vec, eig=eig, occ=occ, mu=mu, rho_r=rho,
                    n_iter=jnp.asarray(it), residual=jnp.asarray(res))


def _make_device_scf(static: Static):
    """Build a jitted, device-resident simplified-Broyden SCF driver.

    Formula follows ABACUS' simplified modified Broyden (Johnson PRB 38,
    12807): ``F = n_out - n_in``, ``dF_i = F_{i-1} - F_i``,
    ``dn_i = n_in_{i-1} - n_in_i``, solve ``<dF_i,dF_j> gamma_j = <dF_i,F>``,
    then ``n_next = n_in + beta F - sum_i gamma_i (dn_i + beta dF_i)``.
    Fixed history (m=8), no dynamic shapes. ``geo``/``cache`` are runtime
    arguments so structure optimization reuses the compiled executable.
    """

    m = 8
    # Start with a more aggressive Broyden damping, then switch to the
    # conservative value. This reaches ~9 iterations on Si/Γ while staying
    # stable on the distorted cell.
    beta_hi = 0.9
    switch = 8

    def fwd(geo, cache, rho_r):
        H = _hamiltonian_from_density(geo, static, rho_r, cache=cache)
        eig, vec = eigh_lowest(H, static.nband)
        occ, mu = occupations(eig, static.kweights, static.nelec,
                              static.sigma, static.spin_degeneracy)
        return density_from_psi(static.basis, geo.lattice, vec, occ,
                                static.kweights, static.spin_degeneracy)

    @jax.jit
    def solve(geo, cache, rho0, beta, tol, max_iter):
        def cond(c):
            _rho, _hn, _hf, _count, _prev, res, it = c
            return (res > tol) & (it < max_iter)

        def body(c):
            rho, hn, hf, count, prev, _res, it = c
            beta_eff = jnp.where(it < switch, beta_hi, beta)
            rho_out = fwd(geo, cache, rho)
            F = rho_out - rho
            res = density_residual(static.basis, geo.lattice, rho_out, rho,
                                   static.nelec)
            reset = res > prev * 1.5
            count = jnp.where(reset, 0, count)
            hn = jnp.concatenate([hn[1:], rho[None, :]], axis=0)
            hf = jnp.concatenate([hf[1:], F[None, :]], axis=0)
            count = jnp.minimum(count + 1, m)

            pair_valid = jnp.arange(m - 1) >= (m - count)
            dF = hf[:-1] - hf[1:]
            dn = hn[:-1] - hn[1:]
            dF = jnp.where(pair_valid[:, None], dF, 0.0)
            dn = jnp.where(pair_valid[:, None], dn, 0.0)
            alpha = jnp.einsum("ik,jk->ij", dF, dF)
            alpha = (alpha
                     + jnp.diag(jnp.where(pair_valid, 0.0, 1.0))
                     + 1e-12 * jnp.eye(m - 1))
            cvec = jnp.einsum("ik,k->i", dF, F)
            gamma = jnp.linalg.solve(alpha, cvec)
            gamma = jnp.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
            new = rho + beta_eff * F - jnp.einsum("i,ij->j", gamma,
                                                  dn + beta_eff * dF)
            lin = rho + beta_eff * F
            new = jnp.where(jnp.all(jnp.isfinite(new)), new, lin)
            return new, hn, hf, count, res, res, it + 1

        z = jnp.zeros((m, static.basis.nfft))
        c0 = (rho0, z, z, jnp.asarray(0), jnp.asarray(1e9),
              jnp.asarray(1.0), jnp.asarray(0))
        rho, hn, hf, count, prev, res, it = jax.lax.while_loop(
            cond, body, c0)
        res_main = res

        # Four raw fixed-point polish steps: the unmixed map need not be a
        # contraction, so do not iterate it to convergence.
        def polish(_, c):
            r, _pres = c
            r_new = fwd(geo, cache, r)
            pres = density_residual(static.basis, geo.lattice, r_new, r,
                                    static.nelec)
            return r_new, pres

        rho, res = jax.lax.fori_loop(0, 4, polish,
                                     (rho, jnp.asarray(1.0)))

        H = _hamiltonian_from_density(geo, static, rho, cache=cache)
        eig, vec = eigh_lowest(H, static.nband)
        occ, mu = occupations(eig, static.kweights, static.nelec,
                              static.sigma, static.spin_degeneracy)
        return rho, res_main, it, eig, vec, occ, mu

    return solve


_DEVICE_SCF_CACHE: dict = {}


def _device_scf(static: Static):
    key = id(static)
    hit = _DEVICE_SCF_CACHE.get(key)
    if hit is not None and hit[0] is static:
        return hit[1]
    if len(_DEVICE_SCF_CACHE) > 64:
        _DEVICE_SCF_CACHE.clear()
    fn = _make_device_scf(static)
    _DEVICE_SCF_CACHE[key] = (static, fn)
    return fn


def _solve_scf_device(geo: Cell, static: Static, cache, beta=0.5,
                      tol=1e-10, max_iter=200):
    solve = _device_scf(static)
    rho0 = initial_density(static, geo)
    rho, res, it, eig, vec, occ, mu = solve(
        geo, cache, rho0, jnp.asarray(beta), jnp.asarray(tol),
        jnp.asarray(max_iter))
    return SCFState(psi=vec, eig=eig, occ=occ, mu=mu, rho_r=rho,
                    n_iter=it, residual=res)


def solve_scf(geo: Cell, static: Static, alpha=0.9, gg0=0.5,
              tol=1e-10, max_iter=200, verbose=False, mixer="diis",
              beta=0.6, driver="auto"):
    """正向求解。

    默认 ``driver="auto"``：先跑设备端 Kerker 循环（``lax.while_loop``，
    无 Python 迭代、无每步 host sync），若未收敛则回退到 Python DIIS。
    ``driver="python"`` 强制旧路径；``driver="device"`` 只跑设备端。
    """
    cache = build_geometry_cache(geo, static)
    if driver not in ("auto", "device", "python"):
        raise ValueError(f"unknown SCF driver: {driver}")

    # libxc is a host callback and cannot be traced inside the device loop.
    # Use the Python driver unless native XC (or the analytic fallback) is
    # available.
    if driver in ("auto", "device") and xc.USE_LIBXC and \
            getattr(xc, "_NATIVE_XC", None) is None:
        if driver == "device":
            raise RuntimeError(
                "device SCF driver requires native XC or ADDFT_XC=analytic")
        driver = "python"

    if driver in ("auto", "device"):
        state = _solve_scf_device(geo, static, cache, beta=beta,
                                  tol=tol, max_iter=max_iter)
        converged = float(state.residual) < tol
        if converged or driver == "device":
            if verbose:
                print(f"  device scf: it={int(state.n_iter)} "
                      f"drho={float(state.residual):.3e}")
            return state
        if verbose:
            print(f"  device scf did not converge (drho="
                  f"{float(state.residual):.3e}); falling back to DIIS")
        # Fallback uses the conservative DIIS damping (0.5); the device
        # driver's faster 0.7 can be too aggressive on hard cases.
        fallback = _solve_scf_python(
            geo, static, rho0=state.rho_r, alpha=alpha, gg0=gg0, tol=tol,
            max_iter=max_iter, verbose=verbose, mixer="diis", beta=0.5,
            cache=cache)
        return fallback._replace(n_iter=state.n_iter + fallback.n_iter)

    return _solve_scf_python(
        geo, static, alpha=alpha, gg0=gg0, tol=tol, max_iter=max_iter,
        verbose=verbose, mixer=mixer, beta=beta, cache=cache)
