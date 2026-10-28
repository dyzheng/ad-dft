"""Matrix-free Hψ: T + V_loc + V_H + V_xc + V_NL, no dense H.

This is the E1 foundation for Davidson/direct minimization. The reference
implementation intentionally uses existing verified primitives; native
fusion comes in a later step.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..geometry.basis import (Basis, gvectors, psi_to_real, real_to_psi)
from ..geometry.cell import Cell
from ..physics import nonlocal_op
from ..system import Static


def kinetic_apply(basis: Basis, lattice, kpts, psi):
    """T ψ in the G shell. psi: (nk, nband, nG)."""
    out = []
    for ik in range(int(kpts.shape[0])):
        kg = gvectors(basis, lattice, kpts[ik])
        t = 0.5 * jnp.sum(kg * kg, axis=-1)
        out.append(psi[ik] * t)
    return jnp.stack(out, axis=0)


def potential_apply(basis: Basis, lattice, v_r, psi):
    """V(r) ψ via real-space multiplication and two FFTs."""
    psi_r = psi_to_real(basis, lattice, psi)
    return real_to_psi(basis, lattice, psi_r * v_r[None, None, :])


def hpsi(geo: Cell, static: Static, psi, v_r, cache=None):
    """Apply H to psi without constructing H.

    psi: (nk, nband, nG) complex; v_r: (nfft,) real effective potential.
    """
    basis = static.basis
    t = kinetic_apply(basis, geo.lattice, static.kpts, psi)
    v = potential_apply(basis, geo.lattice, v_r, psi)
    if cache is None:
        proj = nonlocal_op.projectors(static.psp, static.spec_idx,
                                      static.basis, geo, static.kpts)
        d = nonlocal_op.dion_full(static.psp, static.spec_idx)
    else:
        proj, d = cache.proj, cache.dion
    n = nonlocal_op.vnl_apply(proj, d, psi)
    return t + v + n
