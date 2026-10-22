"""ctypes liibxc bridge for LDA (energy-only + vrho + fxc).

`jax.pure_callback` 把 libxc 的 host 计算暴露成 JAX 原语；能量密度用
`custom_vjp` 把 libxc 的 `vrho` 直接作为一阶导，避免对 ctypes 做 AD。
"""

from __future__ import annotations

import ctypes
import ctypes.util

import jax
import jax.numpy as jnp
import numpy as np

_LIB = ctypes.CDLL(ctypes.util.find_library("xc") or "libxc.so")
_LIB.xc_func_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_LIB.xc_func_init.restype = ctypes.c_int
_LIB.xc_func_end.argtypes = [ctypes.c_void_p]
_LIB.xc_lda_exc_vxc.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_LIB.xc_lda_fxc.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
]


def _ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _host_lda(func_id, rho):
    rho = np.ascontiguousarray(rho, dtype=np.float64).reshape(-1)
    rho = np.maximum(rho, 1e-16)
    zk = np.zeros_like(rho)
    v = np.zeros_like(rho)
    buf = ctypes.create_string_buffer(4096)
    if _LIB.xc_func_init(ctypes.cast(buf, ctypes.c_void_p), func_id, 1) != 0:
        raise RuntimeError(f"libxc init failed: {func_id}")
    try:
        _LIB.xc_lda_exc_vxc(
            ctypes.cast(buf, ctypes.c_void_p), rho.size,
            _ptr(rho), _ptr(zk), _ptr(v))
    finally:
        _LIB.xc_func_end(ctypes.cast(buf, ctypes.c_void_p))
    return rho * zk, v


def _host_fxc(func_id, rho):
    rho = np.ascontiguousarray(rho, dtype=np.float64).reshape(-1)
    rho = np.maximum(rho, 1e-16)
    k = np.zeros_like(rho)
    buf = ctypes.create_string_buffer(4096)
    if _LIB.xc_func_init(ctypes.cast(buf, ctypes.c_void_p), func_id, 1) != 0:
        raise RuntimeError(f"libxc init failed: {func_id}")
    try:
        _LIB.xc_lda_fxc(
            ctypes.cast(buf, ctypes.c_void_p), rho.size,
            _ptr(rho), _ptr(k))
    finally:
        _LIB.xc_func_end(ctypes.cast(buf, ctypes.c_void_p))
    return k


def _host_vrho(func_id, rho):
    return _host_lda(func_id, rho)[1]


@jax.custom_vjp
def _vrho(n, func_id):
    shape = jax.ShapeDtypeStruct(n.shape, jnp.float64)
    return jax.pure_callback(lambda x: _host_vrho(func_id, x), shape, n)


def _vrho_fwd(n, func_id):
    shape = jax.ShapeDtypeStruct(n.shape, jnp.float64)
    v = jax.pure_callback(lambda x: _host_vrho(func_id, x), shape, n)
    return v, (n, func_id)


def _vrho_bwd(res, g):
    n, func_id = res
    return g * libxc_lda_kernel(n, func_id), None


_vrho.defvjp(_vrho_fwd, _vrho_bwd)


def libxc_lda_vrho(n, func_id):
    return _vrho(n, func_id)


def libxc_lda_kernel(n, func_id):
    shape = jax.ShapeDtypeStruct(n.shape, jnp.float64)
    return jax.pure_callback(lambda x: _host_fxc(func_id, x), shape, n)


def _eval(n, func_id):
    shape = jax.ShapeDtypeStruct(n.shape, jnp.float64)
    e, v = jax.pure_callback(
        lambda x: _host_lda(func_id, x),
        (shape, shape), n)
    return e, v


@jax.custom_vjp
def _energy(n, func_id):
    return _eval(n, func_id)[0]


def _energy_fwd(n, func_id):
    e, v = _eval(n, func_id)
    return e, (v,)


def _energy_bwd(res, g):
    (v,) = res
    return g * v, None


_energy.defvjp(_energy_fwd, _energy_bwd)


def libxc_lda_energy_density(n, func_id):
    return _energy(n, func_id)
