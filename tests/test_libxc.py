import jax
import jax.numpy as jnp
import pytest

from addft.physics.libxc_bridge import libxc_lda_energy_density, libxc_lda_kernel


def test_lda_x_energy_density_matches_analytic():
    n = jnp.array([0.2, 1.0, 3.0])
    e = libxc_lda_energy_density(n, 1)  # XC_LDA_X
    eps_x = -(3.0 / 4.0) * (3.0 * n / jnp.pi) ** (1.0 / 3.0)
    assert jnp.allclose(e, n * eps_x, rtol=1e-12, atol=1e-14)


def test_lda_vxc_and_kernel_match_fd():
    n = jnp.array([0.2, 1.0, 3.0])
    e = lambda x: jnp.sum(libxc_lda_energy_density(x, 9))  # PZ correlation
    v = jax.grad(e)(n)
    h = 1e-6
    eye = jnp.eye(n.size)
    v_fd = jnp.stack([(e(n + h * eye[i]) - e(n - h * eye[i])) / (2 * h)
                      for i in range(n.size)])
    assert jnp.allclose(v, v_fd, rtol=1e-6, atol=1e-8)
    k = libxc_lda_kernel(n, 9)
    k_fd = jnp.stack([(jax.grad(e)(n + h * eye[i]) -
                       jax.grad(e)(n - h * eye[i])) / (2 * h)
                      for i in range(n.size)])
    assert jnp.allclose(k, jnp.diag(k_fd), rtol=1e-5, atol=1e-7)
