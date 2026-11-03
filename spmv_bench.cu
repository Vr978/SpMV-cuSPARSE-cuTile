// spmv_bench.cu: CSR sparse matrix-vector multiply (y = A x, FP32) on NVIDIA GPUs.
//
// Kernels
//   scalar        one thread per row
//   vector<T>     T threads per row (T = 2..32), warp-shuffle reduction       (autotuning sweep)
//   adaptive      rows binned by length, each bin launched with its own kernel variant:
//                 short rows -> few threads, medium rows -> one warp, long rows -> one block
//   cuSPARSE      cusparseSpMV, ALG_DEFAULT and CSR_ALG2
//
// Every kernel is checked against an FP64 reference. Error is reported per row as
//   |y_i - y_ref_i| / (|A||x|)_i   in units of u = 2^-24 (FP32 unit roundoff),
// the componentwise measure used in floating-point error analysis of dot products.
//
// Timing: cold L2 (a large memset before every timed run), CUDA events, median of reps.
//
// Usage:  ./spmv_bench matrix.bin name          (binary CSR written by the notebook)
// Output: CSV-style lines on stdout prefixed with DEVICE, MATRIX, RESULT, ERRBUCKET.

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#define CK(call)                                                                        \
  do {                                                                                  \
    cudaError_t e_ = (call);                                                            \
    if (e_ != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,     \
              __LINE__);                                                                \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)
#define CS(call)                                                                        \
  do {                                                                                  \
    cusparseStatus_t s_ = (call);                                                       \
    if (s_ != CUSPARSE_STATUS_SUCCESS) {                                                \
      fprintf(stderr, "cuSPARSE error %d at %s:%d\n", (int)s_, __FILE__, __LINE__);     \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

static const double kU = 5.9604644775390625e-08;  // 2^-24

// ------------------------------------------------------------------------------ kernels

__global__ void spmv_scalar(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                            const float* __restrict__ v, const float* __restrict__ x,
                            float* __restrict__ y) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < n) {
    float s = 0.f;
    for (int k = rp[r]; k < rp[r + 1]; ++k) s += v[k] * __ldg(&x[ci[k]]);
    y[r] = s;
  }
}

// T threads cooperate on one row; LIST = rows come from an index list (adaptive bins).
// No early return: every lane of every warp reaches the shuffles.
template <int T, bool LIST>
__global__ void spmv_vector(int n, const int* __restrict__ rows, const int* __restrict__ rp,
                            const int* __restrict__ ci, const float* __restrict__ v,
                            const float* __restrict__ x, float* __restrict__ y) {
  long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int i = (int)(gid / T);
  int lane = threadIdx.x & (T - 1);
  int r = 0;
  float s = 0.f;
  if (i < n) {
    r = LIST ? rows[i] : i;
    int e = rp[r + 1];
    for (int k = rp[r] + lane; k < e; k += T) s += v[k] * __ldg(&x[ci[k]]);
  }
#pragma unroll
  for (int off = T / 2; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off, T);
  if (lane == 0 && i < n) y[r] = s;
}

// One block per (long) row: per-thread partial sums, then warp shuffles and shared memory.
template <int BS>
__global__ void spmv_block(int n, const int* __restrict__ rows, const int* __restrict__ rp,
                           const int* __restrict__ ci, const float* __restrict__ v,
                           const float* __restrict__ x, float* __restrict__ y) {
  __shared__ float warp_sums[BS / 32];
  int i = blockIdx.x;
  if (i >= n) return;  // uniform across the block
  int r = rows[i];
  int e = rp[r + 1];
  float s = 0.f;
  for (int k = rp[r] + threadIdx.x; k < e; k += BS) s += v[k] * __ldg(&x[ci[k]]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) warp_sums[w] = s;
  __syncthreads();
  if (w == 0) {
    s = (l < BS / 32) ? warp_sums[l] : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if (l == 0) y[r] = s;
  }
}

// FP64 reference and |A||x| for the componentwise error bound.
__global__ void spmv_ref64(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                           const float* __restrict__ v, const float* __restrict__ x,
                           double* __restrict__ y, double* __restrict__ ay) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < n) {
    double s = 0.0, a = 0.0;
    for (int k = rp[r]; k < rp[r + 1]; ++k) {
      double p = (double)v[k] * (double)x[ci[k]];
      s += p;
      a += fabs(p);
    }
    y[r] = s;
    ay[r] = a;
  }
}

template <int T>
static void launch_vec(unsigned grid, int rows, const int* list, const int* rp, const int* ci,
                       const float* v, const float* x, float* y) {
  if (list) spmv_vector<T, true><<<grid, 256>>>(rows, list, rp, ci, v, x, y);
  else      spmv_vector<T, false><<<grid, 256>>>(rows, list, rp, ci, v, x, y);
}

// ------------------------------------------------------------------------------ helpers

struct Csr {
  long long n = 0, m = 0, nnz = 0;
  std::vector<int> rp, ci;
  std::vector<float> v;
};

static Csr read_csr(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  Csr A;
  long long h[3];
  if (fread(h, sizeof(long long), 3, f) != 3) { fprintf(stderr, "bad header\n"); exit(1); }
  A.n = h[0]; A.m = h[1]; A.nnz = h[2];
  A.rp.resize(A.n + 1); A.ci.resize(A.nnz); A.v.resize(A.nnz);
  size_t ok = fread(A.rp.data(), sizeof(int), A.n + 1, f);
  ok += fread(A.ci.data(), sizeof(int), A.nnz, f);
  ok += fread(A.v.data(), sizeof(float), A.nnz, f);
  fclose(f);
  if (ok != (size_t)(A.n + 1 + 2 * A.nnz)) { fprintf(stderr, "truncated file\n"); exit(1); }
  return A;
}

struct Timer {
  cudaEvent_t a, b;
  void* flush = nullptr;
  size_t flush_bytes;
  explicit Timer(size_t fb) : flush_bytes(fb) {
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaMalloc(&flush, flush_bytes));
  }
  // median and min of `reps` cold-L2 runs, in microseconds
  std::pair<double, double> time(const std::function<void()>& run, int reps) {
    run(); run();  // warm-up (also JIT / heuristics inside libraries)
    CK(cudaDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
      CK(cudaMemsetAsync(flush, i & 0xff, flush_bytes));
      CK(cudaEventRecord(a));
      run();
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float ms = 0.f;
      CK(cudaEventElapsedTime(&ms, a, b));
      t.push_back(ms * 1e3);
    }
    std::sort(t.begin(), t.end());
    return {t[t.size() / 2], t[0]};
  }
};

// ------------------------------------------------------------------------------ main

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s matrix.bin name\n", argv[0]); return 1; }
  const char* name = argv[2];
  Csr A = read_csr(argv[1]);
  const int n = (int)A.n, m = (int)A.m;
  const long long nnz = A.nnz;

  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));

  // row-length statistics
  std::vector<int> len(n);
  long long maxlen = 0, empty = 0;
  double mean = (double)nnz / n, var = 0;
  for (int r = 0; r < n; ++r) {
    len[r] = A.rp[r + 1] - A.rp[r];
    maxlen = std::max<long long>(maxlen, len[r]);
    empty += (len[r] == 0);
    var += (len[r] - mean) * (len[r] - mean);
  }
  printf("MATRIX,%s,%d,%d,%lld,%.3f,%lld,%.3f,%lld\n", name, n, m, nnz, mean, maxlen,
         std::sqrt(var / n), empty);

  // deterministic x in [-1, 1)
  std::vector<float> hx(m);
  uint64_t seed = 0x9E3779B97F4A7C15ull;
  for (int j = 0; j < m; ++j) {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    hx[j] = (float)((double)(seed >> 11) / 9007199254740992.0 * 2.0 - 1.0);
  }

  int *d_rp, *d_ci;
  float *d_v, *d_x, *d_y;
  double *d_ref, *d_aref;
  CK(cudaMalloc(&d_rp, sizeof(int) * (n + 1)));
  CK(cudaMalloc(&d_ci, sizeof(int) * std::max<long long>(nnz, 1)));
  CK(cudaMalloc(&d_v, sizeof(float) * std::max<long long>(nnz, 1)));
  CK(cudaMalloc(&d_x, sizeof(float) * m));
  CK(cudaMalloc(&d_y, sizeof(float) * n));
  CK(cudaMalloc(&d_ref, sizeof(double) * n));
  CK(cudaMalloc(&d_aref, sizeof(double) * n));
  CK(cudaMemcpy(d_rp, A.rp.data(), sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_ci, A.ci.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_v, A.v.data(), sizeof(float) * nnz, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_x, hx.data(), sizeof(float) * m, cudaMemcpyHostToDevice));

  // FP64 reference
  spmv_ref64<<<(n + 255) / 256, 256>>>(n, d_rp, d_ci, d_v, d_x, d_ref, d_aref);
  CK(cudaGetLastError());
  std::vector<double> ref(n), aref(n);
  CK(cudaMemcpy(ref.data(), d_ref, sizeof(double) * n, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(aref.data(), d_aref, sizeof(double) * n, cudaMemcpyDeviceToHost));

  // achievable DRAM bandwidth on this GPU: device-to-device copy of 512 MB
  size_t flush_bytes = std::max<size_t>((size_t)prop.l2CacheSize * 4, (size_t)256 << 20);
  Timer timer(flush_bytes);
  double copy_gbs = 0;
  {
    size_t cb = (size_t)256 << 20;
    void *s, *t;
    CK(cudaMalloc(&s, cb)); CK(cudaMalloc(&t, cb));
    auto r = timer.time([&] { CK(cudaMemcpyAsync(t, s, cb, cudaMemcpyDeviceToDevice)); }, 10);
    copy_gbs = 2.0 * cb / (r.first * 1e-6) / 1e9;
    CK(cudaFree(s)); CK(cudaFree(t));
  }
  printf("DEVICE,%s,%d,%d,%d.%d,%.1f\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize,
         prop.major, prop.minor, copy_gbs);

  // compulsory traffic: values + column indices + row pointers + x (once) + y
  const double bytes = 8.0 * nnz + 4.0 * (n + 1) + 4.0 * m + 4.0 * n;

  // adaptive bins by row length
  const int kBins = 6;
  const int lim[kBins] = {4, 8, 16, 32, 2048, 1 << 30};  // inclusive upper bounds
  std::vector<std::vector<int>> bin(kBins);
  for (int r = 0; r < n; ++r) {
    int b = 0;
    while (len[r] > lim[b]) ++b;
    bin[b].push_back(r);
  }
  std::vector<int*> d_bin(kBins, nullptr);
  for (int b = 0; b < kBins; ++b) {
    if (bin[b].empty()) continue;
    CK(cudaMalloc(&d_bin[b], sizeof(int) * bin[b].size()));
    CK(cudaMemcpy(d_bin[b], bin[b].data(), sizeof(int) * bin[b].size(), cudaMemcpyHostToDevice));
  }
  int n_launch_adaptive = 0;
  for (int b = 0; b < kBins; ++b) n_launch_adaptive += !bin[b].empty();

  auto launch_vector = [&](int T, int rows, const int* list) {
    unsigned g = (unsigned)(((long long)rows * T + 255) / 256);
    if (g == 0) return;
    switch (T) {
      case 2:  launch_vec<2>(g, rows, list, d_rp, d_ci, d_v, d_x, d_y); break;
      case 4:  launch_vec<4>(g, rows, list, d_rp, d_ci, d_v, d_x, d_y); break;
      case 8:  launch_vec<8>(g, rows, list, d_rp, d_ci, d_v, d_x, d_y); break;
      case 16: launch_vec<16>(g, rows, list, d_rp, d_ci, d_v, d_x, d_y); break;
      default: launch_vec<32>(g, rows, list, d_rp, d_ci, d_v, d_x, d_y); break;
    }
  };

  struct Variant { std::string name; std::function<void()> run; int launches; };
  std::vector<Variant> variants;
  variants.push_back({"scalar", [&] {
    spmv_scalar<<<(n + 255) / 256, 256>>>(n, d_rp, d_ci, d_v, d_x, d_y); }, 1});
  for (int T : {2, 4, 8, 16, 32})
    variants.push_back({"vector" + std::to_string(T), [&, T] { launch_vector(T, n, nullptr); }, 1});
  variants.push_back({"adaptive", [&] {
    const int tpr[5] = {2, 4, 8, 16, 32};
    for (int b = 0; b < 5; ++b)
      if (!bin[b].empty()) launch_vector(tpr[b], (int)bin[b].size(), d_bin[b]);
    if (!bin[5].empty())
      spmv_block<256><<<(unsigned)bin[5].size(), 256>>>((int)bin[5].size(), d_bin[5], d_rp, d_ci,
                                                        d_v, d_x, d_y);
  }, n_launch_adaptive});

  // cuSPARSE with descriptors and workspace created once, outside the timed region
  cusparseHandle_t h;
  CS(cusparseCreate(&h));
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  CS(cusparseCreateCsr(&matA, n, m, nnz, d_rp, d_ci, d_v, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CS(cusparseCreateDnVec(&vecX, m, d_x, CUDA_R_32F));
  CS(cusparseCreateDnVec(&vecY, n, d_y, CUDA_R_32F));
  const float alpha = 1.f, beta = 0.f;
  struct Alg { const char* name; cusparseSpMVAlg_t alg; void* buf; };
  std::vector<Alg> algs = {{"cusparse_default", CUSPARSE_SPMV_ALG_DEFAULT, nullptr},
                           {"cusparse_csr_alg2", CUSPARSE_SPMV_CSR_ALG2, nullptr}};
  for (auto& a : algs) {
    size_t bs = 0;
    CS(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
                               vecY, CUDA_R_32F, a.alg, &bs));
    CK(cudaMalloc(&a.buf, std::max<size_t>(bs, 16)));
    Alg* ap = &a;
    variants.push_back({a.name, [&, ap] {
      CS(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
                      CUDA_R_32F, ap->alg, ap->buf));
    }, 1});
  }

  // run, time and check every variant
  std::vector<float> hy(n);
  for (auto& var : variants) {
    CK(cudaMemset(d_y, 0xFF, sizeof(float) * n));  // NaN: any row a kernel forgets shows up
    var.run();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hy.data(), d_y, sizeof(float) * n, cudaMemcpyDeviceToHost));

    double max_err = 0, sum_err = 0;
    long long bad = 0;
    const int kB = 32;
    std::vector<double> bmax(kB, 0.0);
    std::vector<long long> bcnt(kB, 0);
    for (int r = 0; r < n; ++r) {
      double e;
      if (!std::isfinite(hy[r])) { e = INFINITY; ++bad; }
      else if (aref[r] == 0.0) e = (hy[r] == 0.f) ? 0.0 : INFINITY;
      else e = std::fabs((double)hy[r] - ref[r]) / aref[r] / kU;
      max_err = std::max(max_err, e);
      if (std::isfinite(e)) sum_err += e;
      int b = 0;
      while ((1 << (b + 1)) <= std::max(len[r], 1) && b < kB - 1) ++b;  // floor(log2(len))
      bmax[b] = std::max(bmax[b], e);
      ++bcnt[b];
    }

    int reps = 30;
    auto t = timer.time(var.run, reps);
    double us = t.first;
    printf("RESULT,%s,%s,%.3f,%.3f,%.3f,%.2f,%.4g,%.4g,%d,%lld\n", name, var.name.c_str(), us,
           t.second, 2.0 * nnz / (us * 1e-6) / 1e9, bytes / (us * 1e-6) / 1e9, max_err,
           sum_err / n, var.launches, bad);
    if (var.name == "scalar" || var.name == "vector32" || var.name == "adaptive" ||
        var.name == "cusparse_default")
      for (int b = 0; b < kB; ++b)
        if (bcnt[b]) printf("ERRBUCKET,%s,%s,%d,%lld,%.4g\n", name, var.name.c_str(), b, bcnt[b],
                            bmax[b]);
    fflush(stdout);
  }

  CS(cusparseDestroySpMat(matA));
  CS(cusparseDestroyDnVec(vecX));
  CS(cusparseDestroyDnVec(vecY));
  CS(cusparseDestroy(h));
  return 0;
}
