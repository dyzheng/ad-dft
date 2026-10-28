# Native CUDA kernels

This directory contains the first native CUDA kernel and its build script.
The kernel is called from JAX through the typed FFI API
(`xla/ffi/api/ffi.h`, shipped in `jaxlib/include`), so it runs on the same
stream as the surrounding JAX computation and participates in JAX tracing.

## Current kernels

### `xc_lda.cu` — fused LDA exchange-correlation

- one pass over the real-space density grid;
- computes `E_xc = Σ n (ε_x + ε_c)` and `v_xc = dE_xc/dn` together;
- no host callback, no intermediate `eps_x`/`eps_c` array;
- analytic PZ81, matching the JAX `ADDFT_XC=analytic` path to machine
  precision (measured `dv < 5e-16`, `dE/E < 2e-16` on 8000 points);
- FP64 only for now.

The JAX wrapper is `src/addft/kernels/xc_lda.py`. It auto-loads
`native/libaddft_kernels.so` when a CUDA device is present. If the library is
missing, or `ADDFT_NATIVE_XC=0`, `addft` falls back to libxc/JAX.

### `eigh.cu` — batched cuSOLVER eigensolver

`cusolverDnZheevdx` with `CUSOLVER_EIG_RANGE_I` computes only the lowest
`nband` eigenpairs. Input is transposed from JAX row-major to cuSOLVER
column-major on the device; the extraction kernel writes `(nk, nband)` and
`(nk, nband, nG)`. Primal-only by design: `scf_map`'s custom JVP uses
first-order perturbation theory and never differentiates through eigh.

At `nG=341` this is roughly parity with XLA's `jnp.linalg.eigh` (the dense
problem is latency-bound on the 3070 Ti); it is kept as the native entry point
for the future Davidson/LOBPCG replacement. `ADDFT_NATIVE_EIGH=0` disables it.

### `pack.cu` — G-shell scatter/gather

Complex128 scatter/gather for coefficient <-> FFT-grid packing. Forward SCF
uses the native kernels; JVP/VJP use the equivalent JAX scatter/gather via
`custom_jvp`, so gradients and `linear_transpose` remain valid. Response /
adjoint paths temporarily fall back to JAX.

| item | JAX | native |
|---|---:|---:|
| `real_from_g` 20 Ry | 7.3 ms | 0.20 ms |
| `real_from_g` 40 Ry | 29.8 ms | 1.34 ms |
| Hartree 20 Ry | 12.0 ms | 1.5 ms |
| density 40 Ry | 18.6 ms | 0.13 ms |

### Device SCF driver

`src/addft/solver/scf.py` now has a `lax.while_loop`-based Kerker driver
(`driver="auto"`, the default). It runs the whole SCF on device, with the
native XC/eigh calls inside the loop, and falls back to Python DIIS if the
residual does not reach tolerance. This is what removes the per-iteration
Python/host-sync overhead.

## Build

Requirements: CUDA >= 12 (matching the JAX CUDA plugin), `nvcc`, JAX with
the CUDA backend.

```bash
# In this environment the CUDA 12 nvcc lives in the conda prefix; the system
# nvcc may be CUDA 10.1 and is rejected by the script.
NVCC=/root/miniconda/bin/nvcc ./native/build.sh

# Optional: target another architecture
NVCC=/root/miniconda/bin/nvcc CUDA_ARCH=sm_90 ./native/build.sh
```

The build outputs `native/libaddft_kernels.so` (ignored by git).

## Use / verify

```bash
export PYTHONPATH=src:tests
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m pytest tests/test_native_xc.py -q
```

The kernel can also be called directly:

```python
from addft.kernels.xc_lda import load_xc_lda
f = load_xc_lda()
vxc, exc_raw = f(rho)   # exc_raw is a plain sum, not (Omega/N) * sum
```

## Measured effect (RTX 3070 Ti, WSL2, FP64)

| item | before | after |
|---|---:|---:|
| `xc_potential_r` (n=8000) | ~18.8 ms (libxc host callback) | ~0.18 ms |
| `scf_map` (ecut=10, nG=341) | ~223 ms | ~50 ms |
| `ad.scf` (ecut=10, 84 iters) | ~33–47 s | ~1.5 s |
| `bench/adjoint.py 729` SCF | 32.94 s | 0.29 s |
| `bench/adjoint.py 729` SCF+grad | 68.58 s | 2.97 s |
| `bench/adjoint.py 3375` SCF | 19.35 s | 0.34 s |
| `bench/adjoint.py 3375` SCF+grad | 41.75 s | 3.19 s |

Correctness gates: energy/force/stress vs the libxc path agree to
`<4e-15 Ha`, `<4e-16 Ha/Bohr`, `<1e-17 Ha/Bohr^3` on the Si test case.
The non-slow suite and the slow force tests pass on GPU, except the
pre-existing `test_abacus_io.py` failure caused by the missing external path
`/root/abacus-develop/examples/...` (unrelated to this backend).

## Next steps

1. Eigh is now the dominant per-step cost (22 ms at `nG=411`, ~100 ms at
   `nG=1139`); Davidson/LOBPCG with a production-grade rank/locking strategy
   is the next target.
2. Force/stress: fuse the envelope-backward reductions.
3. Hartree/density FFT epilogue fusion for larger boxes.
4. AOT cubins, CUDA Graph capture, multi-GPU/k-point streams, wheel packaging.
