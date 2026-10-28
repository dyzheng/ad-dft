"""Native kernels · XC LDA (energy + potential, fused).

This is the first native CUDA kernel in the repository. It is intentionally
small: it validates the whole path

    CUDA source -> nvcc -> JAX FFI -> JAX trace/AD -> SCF

for one fused primitive. The kernel computes ``E_xc`` and ``v_xc = dE_xc/dn``
in a single pass over the density grid (no host callback, no intermediate
array for ``eps_x``/``eps_c``).

Build:

    NVCC=/path/to/nvcc ./native/build.sh

Use:

    from addft.kernels.xc_lda import load_xc_lda
    f = load_xc_lda()
    vxc, exc = f(rho)      # exc is the raw sum n*(eps_x+eps_c), no Omega/N

If the shared library or a CUDA device is unavailable, ``load_xc_lda()``
returns ``None`` and the JAX/libxc path remains in use.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct

_TARGET = "addft_xc_lda"
_LOADED = False
_CALLS: dict = {}


def _lib_path() -> Path:
    env = os.environ.get("ADDFT_NATIVE_LIB")
    if env:
        return Path(env)
    # src/addft/kernels/xc_lda.py -> repository root
    return Path(__file__).resolve().parents[3] / "native" / "libaddft_kernels.so"


def _gpu_available() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def load_xc_lda() -> Optional[Callable]:
    """Return ``f(rho) -> (vxc, exc)`` or ``None`` if unavailable.

    The returned function is an eager JAX FFI call. Direct eager dispatch is
    faster than wrapping it in ``jax.jit`` for these small buffers (measured
    ~0.18 ms vs ~1.2 ms at n=8000). For a full SCF graph, it will be captured
    by the graph/runtime layer later.
    """
    global _LOADED
    if os.environ.get("ADDFT_NATIVE_XC", "1") == "0":
        return None
    if not _gpu_available():
        return None
    path = _lib_path()
    if not path.exists():
        return None

    lib = ctypes.CDLL(str(path))
    try:
        handler = lib.XcLdaHandler
    except AttributeError:
        return None

    if not _LOADED:
        try:
            jax.ffi.register_ffi_target(
                _TARGET, jax.ffi.pycapsule(handler), platform="CUDA")
        except Exception:
            # Another import may already have registered this target. If the
            # target is genuinely missing, the first call below will raise.
            pass
        _LOADED = True

    def call(rho):
        n = int(rho.shape[0])
        key = (n, rho.dtype)
        fn = _CALLS.get(key)
        if fn is None:
            fn = jax.ffi.ffi_call(
                _TARGET,
                [ShapeDtypeStruct((n,), jnp.float64),
                 ShapeDtypeStruct((), jnp.float64)],
            )
            _CALLS[key] = fn
        return fn(rho)

    return call
