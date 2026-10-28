#!/usr/bin/env bash
# Build native kernels as a shared library for JAX FFI.
#
# Usage:
#   ./native/build.sh
#   NVCC=/root/miniconda/bin/nvcc CUDA_ARCH=sm_86 ./native/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/native/libaddft_kernels.so"

PYTHON="${PYTHON:-python}"
JAX_INCLUDE="$("${PYTHON}" -c 'import jax.ffi; print(jax.ffi.include_dir())')"

NVCC="${NVCC:-}"
if [[ -z "${NVCC}" ]]; then
  for cand in /root/miniconda/bin/nvcc /usr/local/cuda/bin/nvcc "$(command -v nvcc || true)"; do
    if [[ -n "${cand}" && -x "${cand}" ]]; then NVCC="${cand}"; break; fi
  done
fi
if [[ -z "${NVCC}" ]]; then
  echo "error: nvcc not found; set NVCC=/path/to/nvcc (CUDA >= 12)" >&2
  exit 1
fi
if ! "${NVCC}" --version | grep -qE 'release (1[2-9]|[2-9][0-9])\.'; then
  echo "error: CUDA >= 12 required by the JAX CUDA plugin; got:" >&2
  "${NVCC}" --version | tail -2 >&2
  exit 1
fi

NVCC_ROOT="$(cd "$(dirname "${NVCC}")/.." && pwd)"

CUDA_INCLUDE="${CUDA_INCLUDE:-}"
if [[ -z "${CUDA_INCLUDE}" ]]; then
  # Prefer the toolkit that owns the selected nvcc. This avoids picking an
  # older system CUDA (e.g. 10.1) that is incompatible with the CUDA 12 JAX
  # plugin when CUDA_HOME points at it.
  for cand in \
    "${NVCC_ROOT}/targets/x86_64-linux/include" \
    "${NVCC_ROOT}/include" \
    "${CONDA_PREFIX:-}/targets/x86_64-linux/include" \
    "${ROOT}/../miniconda/targets/x86_64-linux/include" \
    "${CUDA_HOME:-}/include" \
    /usr/local/cuda/include; do
    if [[ -n "${cand}" && -f "${cand}/cuda_runtime.h" ]]; then CUDA_INCLUDE="${cand}"; break; fi
  done
fi
if [[ -z "${CUDA_INCLUDE}" ]]; then
  echo "error: cuda_runtime.h not found; set CUDA_INCLUDE" >&2
  exit 1
fi

CUDA_LIB="${CUDA_LIB:-}"
if [[ -z "${CUDA_LIB}" ]]; then
  for cand in \
    "${NVCC_ROOT}/targets/x86_64-linux/lib" \
    "${NVCC_ROOT}/lib64" \
    "${CONDA_PREFIX:-}/targets/x86_64-linux/lib" \
    "${ROOT}/../miniconda/targets/x86_64-linux/lib" \
    "${CUDA_HOME:-}/lib64" \
    /usr/local/cuda/lib64; do
    if [[ -n "${cand}" && -e "${cand}/libcudart.so" ]]; then CUDA_LIB="${cand}"; break; fi
  done
fi
if [[ -z "${CUDA_LIB}" ]]; then
  echo "error: libcudart.so not found; set CUDA_LIB" >&2
  exit 1
fi

ARCH="${CUDA_ARCH:-sm_86}"

echo "nvcc:       ${NVCC}"
echo "jax inc:    ${JAX_INCLUDE}"
echo "cuda inc:   ${CUDA_INCLUDE}"
echo "cuda lib:   ${CUDA_LIB}"
echo "arch:       ${ARCH}"

"${NVCC}" -std=c++17 -O3 -shared -Xcompiler -fPIC \
  -arch="${ARCH}" \
  -I"${JAX_INCLUDE}" -I"${CUDA_INCLUDE}" \
  -L"${CUDA_LIB}" -lcudart -lcusolver -lcublas \
  -Xlinker -rpath -Xlinker "${CUDA_LIB}" \
  "${ROOT}/native/xc_lda.cu" "${ROOT}/native/eigh.cu" "${ROOT}/native/pack.cu" \
  -o "${OUT}"

echo "built: ${OUT}"
