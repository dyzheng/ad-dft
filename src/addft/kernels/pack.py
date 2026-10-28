"""Native complex128 G-shell <-> FFT-grid scatter/gather.

JAX's generic scatter/gather for ~10k indices is the dominant cost in
``real_from_g``/``grid_to_coeffs``. These two kernels replace that path.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct

_SCATTER = "addft_scatter_c128"
_GATHER = "addft_gather_c128"
_LOADED = False
_CALLS: dict = {}
_PACK = None
_ENABLED = True


def set_native_pack(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def native_pack_enabled() -> bool:
    return _ENABLED


@contextmanager
def native_pack(enabled: bool):
    """Temporarily enable/disable native pack, restoring on exceptions."""
    global _ENABLED
    prev = _ENABLED
    _ENABLED = bool(enabled)
    try:
        yield
    finally:
        _ENABLED = prev


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


def load_pack() -> Optional[tuple[Callable, Callable]]:
    """Return ``(scatter, gather)`` for complex128 buffers, or ``None``."""
    global _LOADED
    if os.environ.get("ADDFT_NATIVE_PACK", "1") == "0":
        return None
    if not _gpu_available():
        return None
    path = _lib_path()
    if not path.exists():
        return None
    lib = ctypes.CDLL(str(path))
    try:
        s_handler = lib.ScatterC128Handler
        g_handler = lib.GatherC128Handler
    except AttributeError:
        return None

    if not _LOADED:
        for name, handler in ((_SCATTER, s_handler), (_GATHER, g_handler)):
            try:
                jax.ffi.register_ffi_target(
                    name, jax.ffi.pycapsule(handler), platform="CUDA")
            except Exception:
                pass
        _LOADED = True

    def gather(grid, idx, nG):
        grid = jnp.asarray(grid)
        idx = jnp.asarray(idx, dtype=jnp.int32)
        batch, nfft = grid.shape
        nG = int(nG)
        key = ("g", int(batch), int(nfft), nG)
        fn = _CALLS.get(key)
        if fn is None:
            fn = jax.ffi.ffi_call(
                _GATHER,
                [ShapeDtypeStruct((batch, nG), jnp.complex128)])
            _CALLS[key] = fn
        return fn(grid, idx)[0]

    def scatter_n(coeffs, idx, nfft):
        coeffs = jnp.asarray(coeffs)
        idx = jnp.asarray(idx, dtype=jnp.int32)
        batch, nG = coeffs.shape
        nfft = int(nfft)
        key = ("s", int(batch), int(nG), nfft)
        fn = _CALLS.get(key)
        if fn is None:
            fn = jax.ffi.ffi_call(
                _SCATTER,
                [ShapeDtypeStruct((batch, nfft), jnp.complex128)])
            _CALLS[key] = fn
        return fn(coeffs, idx)[0]

    global _PACK
    _PACK = (scatter_n, gather)
    return scatter_n, gather


def _jax_scatter(c, idx, nfft):
    lead = c.shape[:-1]
    flat = jnp.zeros(lead + (nfft,), dtype=c.dtype)
    return flat.at[..., idx].set(c, unique_indices=True)


def _jax_gather(g, idx, nG):
    return jnp.take(g, idx, axis=-1)


def _get_pack():
    global _PACK
    if _PACK is None:
        _PACK = load_pack()
    return _PACK


def native_scatter(c, idx, nfft):
    """Forward native scatter; JVP/VJP use the JAX scatter (so AD works)."""
    pack = _get_pack() if _ENABLED else None
    if pack is None:
        return _jax_scatter(c, idx, nfft)
    raw_s, _ = pack
    nfft = int(nfft)

    @jax.custom_jvp
    def f(x):
        lead = x.shape[:-1]
        out = raw_s(x.reshape(-1, x.shape[-1]), idx, nfft)
        return out.reshape(lead + (nfft,))

    @f.defjvp
    def f_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        return f(x), _jax_scatter(dx, idx, nfft)

    return f(c)


def native_gather(g, idx, nG):
    """Forward native gather; JVP/VJP use the JAX gather (so AD works)."""
    pack = _get_pack() if _ENABLED else None
    if pack is None:
        return _jax_gather(g, idx, nG)
    _, raw_g = pack
    nG = int(nG)

    @jax.custom_jvp
    def f(x):
        lead = x.shape[:-1]
        out = raw_g(x.reshape(-1, x.shape[-1]), idx, nG)
        return out.reshape(lead + (nG,))

    @f.defjvp
    def f_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        return f(x), _jax_gather(dx, idx, nG)

    return f(g)
