"""Native batched Hermitian eigensolver (lowest ``nband`` states).

Primal-only by design: the differentiable path in ``scf_map`` uses analytic
first-order perturbation theory and never differentiates through eigh.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct

_TARGET = "addft_eigh_lowest"
_LOADED = False
_CALLS: dict = {}


def _lib_path() -> Path:
    env = os.environ.get("ADDFT_NATIVE_LIB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "native" / "libaddft_kernels.so"


def _gpu_available() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def load_eigh_lowest() -> Optional[Callable]:
    """Return ``f(H, nband) -> (eig, vec)`` or ``None`` if unavailable."""
    global _LOADED
    if os.environ.get("ADDFT_NATIVE_EIGH", "1") == "0":
        return None
    if not _gpu_available():
        return None
    path = _lib_path()
    if not path.exists():
        return None

    lib = ctypes.CDLL(str(path))
    try:
        handler = lib.EighLowestHandler
    except AttributeError:
        return None

    if not _LOADED:
        try:
            jax.ffi.register_ffi_target(
                _TARGET, jax.ffi.pycapsule(handler), platform="CUDA")
        except Exception:
            pass
        _LOADED = True

    def call(H, nband: int):
        nk, nG, _ = H.shape
        key = (int(nk), int(nG), int(nband), H.dtype)
        fn = _CALLS.get(key)
        if fn is None:
            fn = jax.ffi.ffi_call(
                _TARGET,
                [ShapeDtypeStruct((nk, nband), jnp.float64),
                 ShapeDtypeStruct((nk, nband, nG), jnp.complex128)],
            )
            _CALLS[key] = fn
        return fn(H)

    return call
