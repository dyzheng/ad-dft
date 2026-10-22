import jax.numpy as jnp

from addft.geometry.basis import make_basis
from addft.physics import hartree, xc


def test_ueg_lda_energy():
    basis = make_basis(jnp.eye(3) * 6.0, 2.0)
    n0 = 0.1
    rho = jnp.full((basis.nfft,), n0)
    e = xc.xc_energy_lda(basis, jnp.eye(3) * 6.0, rho)
    omega = 6.0 ** 3
    e_ref = omega * float(xc.lda_energy_density(jnp.asarray(n0)))
    assert abs(float(e) - e_ref) / abs(e_ref) < 1e-6


def test_hartree_planewave_exact_is_the_m0_oracle():
    """I-2 的无截断 oracle：n=1+Σa cos(Gr) 的 V_H 有闭式解。"""
    from addft.geometry.basis import gvectors

    basis = make_basis(jnp.eye(3) * 6.0, 4.0)
    g = gvectors(basis, jnp.eye(3) * 6.0)
    shape = tuple(basis.fft_shape)
    idx = jnp.stack(jnp.meshgrid(*(jnp.arange(n) for n in shape),
                                 indexing="ij"), axis=-1).reshape(-1, 3)
    r = (idx / jnp.array(shape, dtype=jnp.float64)) @ (jnp.eye(3) * 6.0)
    rho = jnp.ones(basis.nfft)
    exact = jnp.zeros(basis.nfft)
    for amp, i in ((0.7, 1), (-0.4, 4), (0.25, 10)):
        g2i = float(jnp.sum(g[i] * g[i]))
        rho = rho + amp * jnp.cos(r @ g[i])
        exact = exact + 4.0 * jnp.pi * amp / g2i * jnp.cos(r @ g[i])
    v = hartree.hartree_potential_r(basis, jnp.eye(3) * 6.0, rho)
    assert float(jnp.max(jnp.abs(v - exact))) < 1e-10
