#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/c_api.h"
#include <cuda_runtime.h>
#include <cmath>

namespace ffi = xla::ffi;

__device__ __forceinline__ double pz_c_eps(double rs) {
  constexpr double A = 0.0311, B = -0.048, C = 0.0020, D = -0.0116;
  constexpr double G = -0.1423, B1 = 1.0529, B2 = 0.3334;
  if (rs < 1.0) return A * log(rs) + B + C * rs * log(rs) + D * rs;
  return G / (1.0 + B1 * sqrt(rs) + B2 * rs);
}

__device__ __forceinline__ double pz_c_v(double n, double rs, double eps_c) {
  constexpr double A = 0.0311, C = 0.0020, D = -0.0116;
  constexpr double G = -0.1423, B1 = 1.0529, B2 = 0.3334;
  double deps;
  if (rs < 1.0) {
    deps = A / rs + C * (log(rs) + 1.0) + D;
  } else {
    double den = 1.0 + B1 * sqrt(rs) + B2 * rs;
    deps = -G * (0.5 * B1 / sqrt(rs) + B2) / (den * den);
  }
  return eps_c - (rs / 3.0) * deps;
}

__global__ void xc_lda_kernel(const double* __restrict__ rho,
                              double* __restrict__ vxc,
                              double* __restrict__ exc,
                              int n) {
  constexpr double PI = 3.14159265358979323846;
  double acc = 0.0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    double nval = fmax(rho[i], 1e-14);
    double rs = cbrt(3.0 / (4.0 * PI * nval));
    double eps_x = -0.75 * cbrt(3.0 * nval / PI);
    double eps_c = pz_c_eps(rs);
    double vx = (4.0 / 3.0) * eps_x;
    double vc = pz_c_v(nval, rs, eps_c);
    vxc[i] = vx + vc;
    acc += nval * (eps_x + eps_c);
  }
  // block reduction
  __shared__ double s[256];
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(exc, s[0]);
}

static ffi::Error XcLda(
    ffi::BufferR1<ffi::F64> rho,
    ffi::ResultBufferR1<ffi::F64> vxc,
    ffi::ResultBufferR0<ffi::F64> exc,
    cudaStream_t stream) {
  const int n = static_cast<int>(rho.element_count());
  cudaMemsetAsync(exc->typed_data(), 0, sizeof(double), stream);
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  xc_lda_kernel<<<blocks, threads, 0, stream>>>(
      rho.typed_data(), vxc->typed_data(), exc->typed_data(), n);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(XcLdaHandler, XcLda,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::BufferR0<ffi::F64>>()
        .Ctx<ffi::PlatformStream<cudaStream_t>>());
