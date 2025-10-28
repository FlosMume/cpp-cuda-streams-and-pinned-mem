#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

// Simple compute: y = a*x + b with extra flops per element to amplify kernel time
__global__ void saxpy_heavy(const float* __restrict__ x,
                            const float* __restrict__ y,
                            float* __restrict__ z,
                            int n,
                            float a, float b, int flop_iters)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float vx = x[i], vy = y[i];
    // Burn flops
    #pragma unroll 4
    for (int k = 0; k <  flop_iters; ++k) {
      vx = a * vx + b;
      vy = a * vy + b;
    }
    z[i] = vx + vy;
  }
}

// Utility: get env var or default
static inline int getenv_int(const char* name, int defv) {
  if (const char* s = std::getenv(name)) {
    try { return std::max(1, std::stoi(s)); } catch (...) { return defv; }
  }
  return defv;
}

int main() {
  // Tunables
  const int N = getenv_int("N", 1<<24);             // total elements
  const int N_STREAMS = getenv_int("N_STREAMS", 4); // number of streams
  const int FLOP_ITERS = getenv_int("FLOP_ITERS", 256);

  printf("Config: N=%d (%.2f MiB/vec)  N_STREAMS=%d  FLOP_ITERS=%d\n",
         N, (N * sizeof(float)) / (1024.0f*1024.0f), N_STREAMS, FLOP_ITERS);

  // Partition into equal-size chunks; last chunk may carry remainder
  int chunk_elems = (N + N_STREAMS - 1) / N_STREAMS;

  // Pinned host allocations enable async DMA
  float *h_x = nullptr, *h_y = nullptr, *h_z = nullptr;
  CHECK_CUDA(cudaMallocHost(&h_x, N * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_y, N * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_z, N * sizeof(float)));

  // Initialize host data
  for (int i = 0; i < N; ++i) {
    h_x[i] = 1.0f;
    h_y[i] = 2.0f;
  }

  // Device buffers sized to one chunk per stream (double-buffering not required here)
  float *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, chunk_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, chunk_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_z, chunk_elems * sizeof(float)));

  // Create streams and events
  std::vector<cudaStream_t> streams(N_STREAMS); // Create a dynamic array (vector) that stores elements of type cudaStream_t
  for (int s = 0; s < N_STREAMS; ++s) CHECK_CUDA(cudaStreamCreate(&streams[s]));

  cudaEvent_t start_all, stop_all;
  CHECK_CUDA(cudaEventCreate(&start_all));
  CHECK_CUDA(cudaEventCreate(&stop_all));

  // Baseline (single stream, synchronous pipeline)
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaEventRecord(start_all));

  const float a = 1.0001f, b = 0.0001f;
  const int block = 256;

  for (int off = 0; off < N; off += chunk_elems) {
    int this_elems = min(chunk_elems, N - off);
    int grid = (this_elems + block - 1) / block;

    CHECK_CUDA(cudaMemcpy(d_x, h_x + off, this_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y + off, this_elems * sizeof(float), cudaMemcpyHostToDevice));

    saxpy_heavy<<<grid, block>>>(d_x, d_y, d_z, this_elems, a, b, FLOP_ITERS);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_z + off, d_z, this_elems * sizeof(float), cudaMemcpyDeviceToHost));
  }

  CHECK_CUDA(cudaEventRecord(stop_all));
  CHECK_CUDA(cudaEventSynchronize(stop_all));
  float ms_baseline = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms_baseline, start_all, stop_all));

  // Validate
  double checksum_base = 0.0;
  for (int i = 0; i < N; ++i) checksum_base += h_z[i];

  // Overlapped pipeline using multiple streams
  CHECK_CUDA(cudaMemset(h_z, 0, N*sizeof(float))); // reuse output

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaEventRecord(start_all));

  for (int off = 0, s = 0; off < N; off += chunk_elems, s = (s + 1) % N_STREAMS) {
    int this_elems = min(chunk_elems, N - off);
    int grid = (this_elems + block - 1) / block;
    cudaStream_t st = streams[s];

    CHECK_CUDA(cudaMemcpyAsync(d_x + s*0, h_x + off, this_elems * sizeof(float), cudaMemcpyHostToDevice, st));
    CHECK_CUDA(cudaMemcpyAsync(d_y + s*0, h_y + off, this_elems * sizeof(float), cudaMemcpyHostToDevice, st));

    saxpy_heavy<<<grid, block, 0, st>>>(d_x + 0, d_y + 0, d_z + 0, this_elems, a, b, FLOP_ITERS);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(h_z + off, d_z + 0, this_elems * sizeof(float), cudaMemcpyDeviceToHost, st));
  }

  // Sync all streams
  for (int s = 0; s < N_STREAMS; ++s) CHECK_CUDA(cudaStreamSynchronize(streams[s]));

  CHECK_CUDA(cudaEventRecord(stop_all));
  CHECK_CUDA(cudaEventSynchronize(stop_all));
  float ms_overlap = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms_overlap, start_all, stop_all));

  // Validate (coarse)
  double checksum_ovl = 0.0;
  for (int i = 0; i < N; ++i) checksum_ovl += h_z[i];

  // Report
  double bytes = 3.0 * N * sizeof(float); // H2D x2 + D2H
  double GB = bytes / 1e9;
  double bw_base = GB / (ms_baseline / 1e3);
  double bw_ovl  = GB / (ms_overlap / 1e3);

  printf("\nBaseline (1 stream):  %.3f ms,  %.2f GB moved → %.2f GB/s  checksum=%.6e\n",
         ms_baseline, GB, bw_base, checksum_base);
  printf("Overlap  (%d streams): %.3f ms,  %.2f GB moved → %.2f GB/s  checksum=%.6e\n",
         N_STREAMS, ms_overlap, GB, bw_ovl, checksum_ovl);
  printf("Speedup: %.2fx\n", ms_baseline / ms_overlap);

  // Cleanup
  for (int s = 0; s < N_STREAMS; ++s) cudaStreamDestroy(streams[s]);
  cudaEventDestroy(start_all);
  cudaEventDestroy(stop_all);
  cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
  cudaFreeHost(h_x); cudaFreeHost(h_y); cudaFreeHost(h_z);
  return 0;
}
