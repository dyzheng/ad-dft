"""Matrix-free preconditioned block Davidson for the lowest nband states.

Aligned with ABACUS `Diago_DavSubspace` in the important engineering details:
  * subspace size ~ 4*nband;
  * Hψ is cached and only new columns are applied;
  * ABACUS-style diagonal preconditioner |h_diag - λ|;
  * restart to the current nband Ritz vectors when the subspace is full.
"""

from __future__ import annotations

import jax.numpy as jnp


def _orth(x):
    q, _ = jnp.linalg.qr(x)
    return q


def _precond(lam, diag, kind="abacus", alpha=1.0):
    if kind == "identity":
        return jnp.ones_like(lam[:, None, :])
    if kind == "kinetic":
        den = diag[:, :, None] - lam[:, None, :]
        return jnp.where(jnp.abs(den) > 1e-6, den, 1.0)
    # ABACUS precondition_op: 0.5*(1+x+sqrt(1+(x-1)^2)), x=|h_diag-λ|
    x = jnp.abs(diag[:, :, None] - lam[:, None, :])
    base = 0.5 * (1.0 + x + jnp.sqrt(1.0 + (x - 1.0) ** 2))
    return jnp.power(jnp.maximum(base, 1e-6), alpha)


def davidson_lowest(apply, x0, nband: int, diag, max_sub: int = 24,
                    max_iter: int = 40, precond: str = "abacus",
                    precond_alpha: float = 1.0):
    """Return (eig, vec, residual).

    apply: ``(nk, nG, nb) -> (nk, nG, nb)``.
    x0:    ``(nk, nG, nb)`` initial guess (warm start).
    diag:  ``(nk, nG)`` h_diag (kinetic + v_loc(G=0)).
    vec:   ``(nk, nb, nG)`` matching ``eigh_lowest``.
    """
    nb = int(nband)
    q = _orth(x0)
    hq = apply(q)
    nbas = nb
    lam = jnp.zeros((q.shape[0], nb), dtype=jnp.float64)
    for _ in range(int(max_iter)):
        t = jnp.matmul(jnp.conj(jnp.swapaxes(q, -1, -2)), hq)
        t = 0.5 * (t + jnp.conj(jnp.swapaxes(t, -1, -2)))
        w, u = jnp.linalg.eigh(t)
        q = jnp.matmul(q, u)
        hq = jnp.matmul(hq, u)
        x = q[:, :, :nb]
        hx = hq[:, :, :nb]
        lam = w[:, :nb]
        r = hx - x * lam[:, None, :]
        r = r / _precond(lam, diag, precond, precond_alpha)
        # Full reorthogonalization against the current subspace (twice).
        for _ in range(2):
            r = r - jnp.matmul(
                q, jnp.matmul(jnp.conj(jnp.swapaxes(q, -1, -2)), r))
        qadd, _ = jnp.linalg.qr(r)
        q = jnp.concatenate([q, qadd], axis=2)
        hq = jnp.concatenate([hq, apply(qadd)], axis=2)
        nbas += nb
        if nbas > int(max_sub):
            q = x
            hq = hx
            nbas = nb
    t = jnp.matmul(jnp.conj(jnp.swapaxes(q, -1, -2)), hq)
    t = 0.5 * (t + jnp.conj(jnp.swapaxes(t, -1, -2)))
    w, u = jnp.linalg.eigh(t)
    x = jnp.matmul(q, u[:, :, :nb])
    lam = w[:, :nb]
    r = jnp.matmul(hq, u[:, :, :nb]) - x * lam[:, None, :]
    res = jnp.max(jnp.linalg.norm(r, axis=1))
    return lam, jnp.swapaxes(x, -1, -2), res
