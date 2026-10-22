"""实球谐，l≤2，顺序与 ABACUS YlmReal 一致：每个 l 为 m=0,+1,-1,+2,-2..."""

import jax.numpy as jnp

_FOUR_PI = 4.0 * jnp.pi


def ylm_real(lmax: int, g):
    """g: (...,3) 笛卡尔；返回 (...,(lmax+1)^2)。"""
    g = jnp.asarray(g)
    x, y, z = g[..., 0], g[..., 1], g[..., 2]
    r2 = jnp.sum(g * g, axis=-1)
    rs = jnp.sqrt(jnp.where(r2 > 1e-30, r2, 1.0))
    out = [jnp.ones_like(rs) / jnp.sqrt(_FOUR_PI)]
    if lmax >= 1:
        c1 = jnp.sqrt(3.0 / _FOUR_PI) / rs
        out += [c1 * z, -c1 * x, -c1 * y]
    if lmax >= 2:
        out += [
            jnp.sqrt(5.0 / _FOUR_PI) * (3.0 * z * z - r2) / (2.0 * rs * rs),
            -jnp.sqrt(15.0 / _FOUR_PI) * x * z / (rs * rs),
            -jnp.sqrt(15.0 / _FOUR_PI) * y * z / (rs * rs),
            jnp.sqrt(15.0 / (16.0 * jnp.pi)) * (x * x - y * y) / (rs * rs),
            jnp.sqrt(15.0 / _FOUR_PI) * x * y / (rs * rs),
        ]
    return jnp.stack(out, axis=-1)
