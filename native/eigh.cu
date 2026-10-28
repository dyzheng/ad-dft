// Native batched Hermitian eigensolver for the lowest `nband` states.
//
// JAX's generic `jnp.linalg.eigh` costs ~18 ms per SCF iteration on the
// nG=341 test case. Here we call cuSOLVER `zheevd` per k-point directly on
// the XLA stream, then extract only the lowest nband eigenvectors with a
// small kernel. The eigensolver is primal-only: the existing custom JVP in
// `scf_map` uses first-order perturbation theory and never differentiates
// through this call.

#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <complex>
#include <mutex>

namespace ffi = xla::ffi;

using C128 = std::complex<double>;

__global__ void transpose_copy_kernel(const cuDoubleComplex* __restrict__ src,
                                      cuDoubleComplex* __restrict__ dst,
                                      int n, int nk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nk * n * n;
  if (idx >= total) return;
  int k = idx / (n * n);
  int rem = idx - k * n * n;
  int row = rem / n;
  int col = rem - row * n;
  dst[k * n * n + col * n + row] = src[k * n * n + row * n + col];
}

__global__ void extract_kernel(const cuDoubleComplex* __restrict__ A,
                               const double* __restrict__ W,
                               double* __restrict__ eig,
                               cuDoubleComplex* __restrict__ vec,
                               int n, int nband, int nk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = nk * nband * n;
  if (idx >= total) return;
  int k = idx / (nband * n);
  int rem = idx - k * nband * n;
  int i = rem / n;
  int j = rem - i * n;
  // A is column-major; eigenvector i is column i.
  vec[(k * nband + i) * n + j] = A[k * n * n + i * n + j];
  if (j == 0) eig[k * nband + i] = W[k * n + i];
}

static cusolverDnHandle_t g_handle = nullptr;
static std::once_flag g_once;

static ffi::Error EighLowest(
    ffi::BufferR3<ffi::C128> H,
    ffi::ResultBufferR2<ffi::F64> eig,
    ffi::ResultBufferR3<ffi::C128> vec,
    cudaStream_t stream) {
  auto hdims = H.dimensions();
  auto edims = eig->dimensions();
  auto vdims = vec->dimensions();
  const int nk = static_cast<int>(hdims[0]);
  const int n = static_cast<int>(hdims[1]);
  const int nband = static_cast<int>(edims[1]);
  if (nband > n) return ffi::Error::InvalidArgument("nband > nG");
  if (vdims[0] != nk || vdims[1] != nband || vdims[2] != n)
    return ffi::Error::InvalidArgument("vec shape mismatch");

  std::call_once(g_once, [] {
    if (cusolverDnCreate(&g_handle) != CUSOLVER_STATUS_SUCCESS) g_handle = nullptr;
  });
  if (g_handle == nullptr) return ffi::Error::Internal("cusolverDnCreate failed");
  cusolverDnSetStream(g_handle, stream);

  const size_t mat_bytes = static_cast<size_t>(nk) * n * n * sizeof(cuDoubleComplex);
  const size_t w_bytes = static_cast<size_t>(nk) * n * sizeof(double);
  const size_t info_bytes = static_cast<size_t>(nk) * sizeof(int);
  int h_meig = 0;  // host scalar, per cuSOLVER syevdx contract

  cuDoubleComplex* Hwork = nullptr;
  double* W = nullptr;
  int* info = nullptr;
  void* work = nullptr;
  int lwork = 0;

  if (cudaMallocAsync(&Hwork, mat_bytes, stream) != cudaSuccess ||
      cudaMallocAsync(&W, w_bytes, stream) != cudaSuccess ||
      cudaMallocAsync(&info, info_bytes, stream) != cudaSuccess) {
    return ffi::Error::Internal("cudaMallocAsync failed");
  }

  // Row-major JAX buffer -> column-major cuSOLVER buffer (H is Hermitian).
  {
    int threads = 256;
    int total = nk * n * n;
    int blocks = (total + threads - 1) / threads;
    transpose_copy_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const cuDoubleComplex*>(H.typed_data()),
        Hwork, n, nk);
  }

  cusolverStatus_t st = cusolverDnZheevdx_bufferSize(
      g_handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
      CUBLAS_FILL_MODE_LOWER, n, Hwork, n,
      0.0, 0.0, 1, nband, &h_meig, W, &lwork);
  if (st != CUSOLVER_STATUS_SUCCESS) {
    cudaFreeAsync(Hwork, stream);
    cudaFreeAsync(W, stream);
    cudaFreeAsync(info, stream);
    return ffi::Error::Internal("zheevdx_bufferSize failed");
  }
  if (cudaMallocAsync(&work, static_cast<size_t>(lwork) * sizeof(cuDoubleComplex),
                      stream) != cudaSuccess) {
    cudaFreeAsync(Hwork, stream);
    cudaFreeAsync(W, stream);
    cudaFreeAsync(info, stream);
    return ffi::Error::Internal("cudaMallocAsync(work) failed");
  }

  for (int k = 0; k < nk; ++k) {
    st = cusolverDnZheevdx(
        g_handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
        CUBLAS_FILL_MODE_LOWER, n, Hwork + static_cast<size_t>(k) * n * n, n,
        0.0, 0.0, 1, nband, &h_meig,
        W + static_cast<size_t>(k) * n,
        reinterpret_cast<cuDoubleComplex*>(work), lwork, info + k);
    if (st != CUSOLVER_STATUS_SUCCESS) {
      cudaFreeAsync(work, stream);
      cudaFreeAsync(Hwork, stream);
      cudaFreeAsync(W, stream);
      cudaFreeAsync(info, stream);
      return ffi::Error::Internal("zheevdx failed");
    }
  }

  {
    int threads = 256;
    int total = nk * nband * n;
    int blocks = (total + threads - 1) / threads;
    extract_kernel<<<blocks, threads, 0, stream>>>(
        Hwork, W, eig->typed_data(),
        reinterpret_cast<cuDoubleComplex*>(vec->typed_data()),
        n, nband, nk);
  }

  // Stream-ordered frees: safe before the kernels complete.
  cudaFreeAsync(work, stream);
  cudaFreeAsync(Hwork, stream);
  cudaFreeAsync(W, stream);
  cudaFreeAsync(info, stream);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EighLowestHandler, EighLowest,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<ffi::BufferR2<ffi::F64>>()
        .Ret<ffi::BufferR3<ffi::C128>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());
