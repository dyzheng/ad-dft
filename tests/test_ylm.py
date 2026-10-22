import jax.numpy as jnp

from addft.physics.ylm import ylm_real


def test_ylm_l1_matches_abacus_order():
    # ABACUS 顺序：Y10, Y11, Y1-1 = +z, -x, -y（乘 √(3/4π)/r）
    g = jnp.array([[0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]])
    y = ylm_real(1, g)
    c = jnp.sqrt(3.0 / (4.0 * jnp.pi))
    assert jnp.allclose(y[:, 0], jnp.ones(3) / jnp.sqrt(4.0 * jnp.pi))
    assert jnp.allclose(y[:, 1], jnp.array([c, 0.0, 0.0]), atol=1e-14)
    assert jnp.allclose(y[:, 2], jnp.array([0.0, -c, 0.0]), atol=1e-14)
    assert jnp.allclose(y[:, 3], jnp.array([0.0, 0.0, -c]), atol=1e-14)


def test_ylm_l2_matches_abacus_order():
    import math
    s5 = math.sqrt(5.0 / (4.0 * math.pi))
    s15 = math.sqrt(15.0 / (4.0 * math.pi))
    s1516 = math.sqrt(15.0 / (16.0 * math.pi))
    g = jnp.array([[0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [1.0, -1.0, 0.0],
                   [1.0, 1.0, 0.0]], dtype=jnp.float64)
    g = g / jnp.linalg.norm(g, axis=-1, keepdims=True)
    y = ylm_real(2, g)
    ref = jnp.array([
        [s5, 0.0, 0.0, 0.0, 0.0],
        [-s5 / 2.0, 0.0, 0.0, s1516, 0.0],
        [-s5 / 2.0, 0.0, 0.0, -s1516, 0.0],
        [-s5 / 2.0, 0.0, 0.0, 0.0, -s15 / 2.0],
        [-s5 / 2.0, 0.0, 0.0, 0.0, s15 / 2.0],
    ])
    assert jnp.allclose(y[:, 4:], ref, atol=1e-14)


def test_projector_count_has_m_channels():
    from addft.geometry.psp import synthetic
    from addft.physics.nonlocal_op import projectors
    from conftest import make_system

    s = make_system()
    psp = synthetic()
    # 合成赝势的 l = [0, 1] → 1 + 3 = 4 个 m 通道
    assert int(psp.nh[0]) == 4
    proj = projectors(psp, s.static.spec_idx, s.static.basis, s.geo, s.static.kpts)
    assert proj.shape[1] == s.geo.natom * 4
