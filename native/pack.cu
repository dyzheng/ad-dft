// Native complex128 scatter/gather for G-shell <-> FFT-grid packing.
//
// JAX lowers `.at[idx].set(..., unique_indices=True)` and `take(idx)` to
// surprisingly slow generic kernels at the 10k-index scale (~25 ms for the
// 40 Ry Si box). These two simple kernels replace that path; the FFT itself
// is only ~0.2 ms.

#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <complex>

namespace ffi = xla::ffi;
using C128 = std::complex<double>;

__global__ void scatter_kernel(const cuDoubleComplex* __restrict__ coeffs,
                               const int* __restrict__ idx,
                               cuDoubleComplex* __restrict__ out,
                               int batch, int nG, int nfft) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * nG;
  if (t >= total) return;
  int b = t / nG;
  int i = t - b * nG;
  out[b * nfft + idx[i]] = coeffs[t];
}

__global__ void gather_kernel(const cuDoubleComplex* __restrict__ grid,
                              const int* __restrict__ idx,
                              cuDoubleComplex* __restrict__ out,
                              int batch, int nG, int nfft) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * nG;
  if (t >= total) return;
  int b = t / nG;
  int i = t - b * nG;
  out[t] = grid[b * nfft + idx[i]];
}

static ffi::Error ScatterC128(
    ffi::BufferR2<ffi::C128> coeffs,
    ffi::BufferR1<ffi::S32> idx,
    ffi::ResultBufferR2<ffi::C128> out,
    cudaStream_t stream) {
  auto cd = coeffs.dimensions();
  auto od = out->dimensions();
  const int batch = static_cast<int>(cd[0]);
  const int nG = static_cast<int>(cd[1]);
  const int nfft = static_cast<int>(od[1]);
  if (static_cast<int>(idx.element_count()) != nG)
    return ffi::Error::InvalidArgument("idx length != nG");
  cudaMemsetAsync(out->typed_data(), 0,
                  static_cast<size_t>(batch) * nfft * sizeof(C128), stream);
  const int threads = 256;
  const int total = batch * nG;
  const int blocks = (total + threads - 1) / threads;
  scatter_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const cuDoubleComplex*>(coeffs.typed_data()),
      idx.typed_data(),
      reinterpret_cast<cuDoubleComplex*>(out->typed_data()),
      batch, nG, nfft);
  return ffi::Error::Success();
}

static ffi::Error GatherC128(
    ffi::BufferR2<ffi::C128> grid,
    ffi::BufferR1<ffi::S32> idx,
    ffi::ResultBufferR2<ffi::C128> out,
    cudaStream_t stream) {
  auto gd = grid.dimensions();
  auto od = out->dimensions();
  const int batch = static_cast<int>(gd[0]);
  const int nfft = static_cast<int>(gd[1]);
  const int nG = static_cast<int>(od[1]);
  if (static_cast<int>(idx.element_count()) != nG)
    return ffi::Error::InvalidArgument("idx length != nG");
  const int threads = 256;
  const int total = batch * nG;
  const int blocks = (total + threads - 1) / threads;
  gather_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const cuDoubleComplex*>(grid.typed_data()),
      idx.typed_data(),
      reinterpret_cast<cuDoubleComplex*>(out->typed_data()),
      batch, nG, nfft);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ScatterC128Handler, ScatterC128,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR2<ffi::C128>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR2<ffi::C128>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(GatherC128Handler, GatherC128,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR2<ffi::C128>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR2<ffi::C128>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());
