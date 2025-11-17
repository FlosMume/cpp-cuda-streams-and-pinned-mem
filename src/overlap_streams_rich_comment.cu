// overlap_streams.cu
//
// Demonstration: overlap host↔device memory copies with GPU compute using
// CUDA streams + pinned (page-locked) host memory.
//
// Key ideas:
//   - Pinned host memory enables true asynchronous DMA transfers
//     (cudaMemcpyAsync can run concurrently with kernels).
//   - Multiple CUDA streams let us pipeline: H2D → compute → D2H
//     across independent chunks of the data.
//   - We compare a baseline (single stream, sequential) pipeline
//     with a multi-stream overlapped version.
//
// This file is meant to be *teachable*, so the comments are intentionally rich.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Error checking macro
//
// Almost every CUDA runtime call returns a cudaError_t. We wrap calls in this
// macro so that if anything fails, we print a useful message and exit.
// -----------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// -----------------------------------------------------------------------------
// Utility: read an integer environment variable, or fall back to a default.
//
// Example usage:
//   int N = getenv_int("N", 1<<24);        // default 16M elements
//   int NS = getenv_int("N_STREAMS", 4);   // default 4 streams
//
// We also clamp the value to be at least 1 to avoid nonsense like N_STREAMS=0.
// -----------------------------------------------------------------------------
static inline int getenv_int(const char* name, int defv)
{
    if (const char* s = std::getenv(name)) {
        try {
            // std::stoi can throw if the string is not a valid integer.
            int v = std::stoi(s);
            if (v < 1) return defv;
            return v;
        } catch (...) {
            // On any parse error, just fall back to the default.
            return defv;
        }
    }
    // Variable not set → use default.
    return defv;
}

// -----------------------------------------------------------------------------
// Compute-heavy kernel: saxpy_heavy
//
// This is a *synthetic* kernel. It is NOT trying to implement a nice formula
// like z[i] = a*x[i] + b. Instead, it intentionally performs a lot of math
// per element to keep the GPU busy long enough that we can clearly see overlap
// between memcpy and compute.
//
// For each valid index i in [0, n):
//   - Load x[i] and y[i] into registers (vx, vy).
//   - Repeat: vx = a*vx + b; vy = a*vy + b;  flop_iters times.
//   - Store z[i] = vx + vy.
//
// The final numerical result is not important; timing is.
// -----------------------------------------------------------------------------
__global__ void saxpy_heavy(const float* __restrict__ x,
                            const float* __restrict__ y,
                            float* __restrict__ z,
                            int n,
                            float a,
                            float b,
                            int flop_iters)
{
    // Compute this thread's 1D global index.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-bounds.
    if (i < n) {
        // Load one element from each input into registers.
        // Using local variables keeps data in fast registers instead of
        // repeatedly reading from global memory.
        float vx = x[i];
        float vy = y[i];

        // ---------------------------------------------------------------------
        // Burn FLOPs:
        //
        // We artificially increase compute work by applying a simple affine
        // transform many times. This makes the kernel "heavier" so that:
        //   - Kernel runtime is not negligible compared to memcpy time.
        //   - Overlap between copies and compute becomes visible.
        //   - The compiler cannot easily optimize away the work, because the
        //     final result is used to write z[i].
        //
        // #pragma unroll asks the compiler to partially unroll the loop,
        // improving ILP (instruction-level parallelism).
        // ---------------------------------------------------------------------
        #pragma unroll 4
        for (int k = 0; k < flop_iters; ++k) {
            vx = a * vx + b;
            vy = a * vy + b;
        }

        // Combine the two transformed values; the exact formula is unimportant.
        // The key is that z[i] depends on both x[i] and y[i] and on all the
        // repeated operations, so the work cannot be considered "dead code".
        z[i] = vx + vy;
    }
}

// -----------------------------------------------------------------------------
// Main program
// -----------------------------------------------------------------------------
int main()
{
    // -------------------------------------------------------------------------
    // 1. Read tunables from the environment
    //
    // N          : total number of elements in each vector
    // N_STREAMS  : how many CUDA streams we use in the overlapped version
    // FLOP_ITERS : how compute-heavy the kernel is
    //
    // You can tweak these at runtime without recompiling, e.g.:
    //   N=8388608 N_STREAMS=8 FLOP_ITERS=512 ./overlap_streams
    // -------------------------------------------------------------------------
    const int N          = getenv_int("N",          1 << 24); // default: 16M
    const int N_STREAMS  = getenv_int("N_STREAMS",  4);       // default: 4
    const int FLOP_ITERS = getenv_int("FLOP_ITERS", 256);     // default: 256

    // Size in bytes of one vector (x or y or z).
    const size_t bytes_per_vec = static_cast<size_t>(N) * sizeof(float);

    // Print configuration in human-readable form (MiB/vec).
    const double mib_per_vec = bytes_per_vec / (1024.0 * 1024.0);
    printf("Config:\n");
    printf("  N          = %d elements\n", N);
    printf("  N_STREAMS  = %d\n", N_STREAMS);
    printf("  FLOP_ITERS = %d\n", FLOP_ITERS);
    printf("  %.2f MiB per vector (x, y, z each)\n\n", mib_per_vec);

    // -------------------------------------------------------------------------
    // 2. Compute a chunk size for the pipeline.
    //
    // We partition the N elements into roughly equal-size chunks so that each
    // host loop iteration can operate on a contiguous subrange [off, off+len).
    //
    // chunk_elems = ceil(N / N_STREAMS)
    // -------------------------------------------------------------------------
    const int chunk_elems = (N + N_STREAMS - 1) / N_STREAMS;

    // -------------------------------------------------------------------------
    // 3. Allocate pinned host memory.
    //
    // We use cudaMallocHost instead of malloc/new. Pinned (page-locked) memory:
    //   - cannot be paged out by the OS
    //   - has stable physical addresses
    //   - is suitable for direct DMA by the GPU
    //
    // This is REQUIRED for true asynchronous cudaMemcpyAsync.
    // -------------------------------------------------------------------------
    float* h_x = nullptr;
    float* h_y = nullptr;
    float* h_z = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_x, bytes_per_vec));
    CHECK_CUDA(cudaMallocHost(&h_y, bytes_per_vec));
    CHECK_CUDA(cudaMallocHost(&h_z, bytes_per_vec));

    // Initialize input data on the host.
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
        h_z[i] = 0.0f; // not strictly required, but keeps things neat
    }

    // -------------------------------------------------------------------------
    // 4. Allocate device memory.
    //
    // Here we allocate full-length device buffers for x, y, z. During the
    // pipelined run, different chunks [off, off+len) of these arrays are used
    // by different streams. Chunks do not overlap in index space, so this is
    // safe for concurrency.
    // -------------------------------------------------------------------------
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, bytes_per_vec));
    CHECK_CUDA(cudaMalloc(&d_y, bytes_per_vec));
    CHECK_CUDA(cudaMalloc(&d_z, bytes_per_vec));

    // -------------------------------------------------------------------------
    // 5. Create CUDA streams and timing events.
    //
    // Each stream represents an independent queue of operations on the GPU.
    // Commands in different streams can run concurrently when the hardware
    // resources (compute + copy engines) allow it.
    // -------------------------------------------------------------------------
    std::vector<cudaStream_t> streams(N_STREAMS);
    for (int s = 0; s < N_STREAMS; ++s) {
        CHECK_CUDA(cudaStreamCreate(&streams[s]));
    }

    cudaEvent_t start_all, stop_all;
    CHECK_CUDA(cudaEventCreate(&start_all));
    CHECK_CUDA(cudaEventCreate(&stop_all));

    const float a = 1.0001f;
    const float b = 0.0001f;
    const int   block = 256; // typical, reasonable block size

    // -------------------------------------------------------------------------
    // 6. Baseline: single-stream, sequential pipeline
    //
    // We still process the data in chunks, but:
    //   - host-device copies are synchronous (cudaMemcpy)
    //   - we use the default stream (implicit 0) for kernel launches
    //   - there is no overlap between H2D, compute, and D2H
    //
    // This gives us a fair baseline to compare against the overlapped version.
    // -------------------------------------------------------------------------
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start_all));

    for (int off = 0; off < N; off += chunk_elems) {
        // How many elements in this chunk? (last chunk may be smaller)
        int this_elems = (off + chunk_elems <= N)
                       ? chunk_elems
                       : (N - off);

        const size_t this_bytes = static_cast<size_t>(this_elems) * sizeof(float);
        const int grid = (this_elems + block - 1) / block;

        // Copies are synchronous here: the host thread blocks until each
        // transfer completes. No overlap with compute is possible.
        CHECK_CUDA(cudaMemcpy(d_x + off, h_x + off, this_bytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_y + off, h_y + off, this_bytes,
                              cudaMemcpyHostToDevice));

        // Launch kernel in the default stream (stream 0).
        saxpy_heavy<<<grid, block>>>(d_x + off,
                                     d_y + off,
                                     d_z + off,
                                     this_elems,
                                     a, b, FLOP_ITERS);
        CHECK_CUDA(cudaGetLastError());

        // Copy results back synchronously.
        CHECK_CUDA(cudaMemcpy(h_z + off, d_z + off, this_bytes,
                              cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaEventRecord(stop_all));
    CHECK_CUDA(cudaEventSynchronize(stop_all));

    float ms_baseline = 0.0f; // elapsed time in milliseconds
    CHECK_CUDA(cudaEventElapsedTime(&ms_baseline, start_all, stop_all));

    // Compute a simple checksum as a correctness sanity check.
    double checksum_base = 0.0;
    for (int i = 0; i < N; ++i) {
        checksum_base += h_z[i];
    }

    // -------------------------------------------------------------------------
    // 7. Overlapped version: multi-stream pipelined pipeline
    //
    // Now we use cudaMemcpyAsync and multiple streams. Each chunk:
    //   - H2D copy on its own stream
    //   - kernel on that stream
    //   - D2H copy on that stream
    //
    // Different chunks live in disjoint index ranges [off, off+this_elems),
    // so they can safely be processed concurrently.
    // -------------------------------------------------------------------------

    // Re-initialize output buffer (good practice for clarity).
    for (int i = 0; i < N; ++i) {
        h_z[i] = 0.0f;
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start_all));

    int stream_index = 0;

    for (int off = 0; off < N; off += chunk_elems) {
        int this_elems = (off + chunk_elems <= N)
                       ? chunk_elems
                       : (N - off);
        const size_t this_bytes = static_cast<size_t>(this_elems) * sizeof(float);
        const int grid = (this_elems + block - 1) / block;

        // Pick a stream in round-robin fashion.
        cudaStream_t st = streams[stream_index];
        stream_index = (stream_index + 1) % N_STREAMS;

        // Asynchronous copies:
        // Since h_x/h_y are pinned, and d_x/d_y are device memory,
        // these calls enqueue DMA operations and return immediately.
        CHECK_CUDA(cudaMemcpyAsync(d_x + off, h_x + off, this_bytes,
                                   cudaMemcpyHostToDevice, st));
        CHECK_CUDA(cudaMemcpyAsync(d_y + off, h_y + off, this_bytes,
                                   cudaMemcpyHostToDevice, st));

        // Launch kernel in this stream. It will run after the H2D copies
        // in that stream complete, but can overlap with work in other streams.
        saxpy_heavy<<<grid, block, 0, st>>>(d_x + off,
                                            d_y + off,
                                            d_z + off,
                                            this_elems,
                                            a, b, FLOP_ITERS);
        CHECK_CUDA(cudaGetLastError());

        // Asynchronous D2H copy of results back to the matching region
        // of h_z. Again, this can overlap with H2D or compute in other streams.
        CHECK_CUDA(cudaMemcpyAsync(h_z + off, d_z + off, this_bytes,
                                   cudaMemcpyDeviceToHost, st));
    }

    // Wait for all streams to finish their queued work.
    for (int s = 0; s < N_STREAMS; ++s) {
        CHECK_CUDA(cudaStreamSynchronize(streams[s]));
    }

    CHECK_CUDA(cudaEventRecord(stop_all));
    CHECK_CUDA(cudaEventSynchronize(stop_all));

    float ms_overlap = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_overlap, start_all, stop_all));

    // Checksum for overlapped result.
    double checksum_ovl = 0.0;
    for (int i = 0; i < N; ++i) {
        checksum_ovl += h_z[i];
    }

    // -------------------------------------------------------------------------
    // 8. Report timings and effective bandwidth.
    //
    // We treat the total data movement per run as:
    //   bytes_moved = (H2D x2 + D2H x1) = 3 * N * sizeof(float)
    //
    // Then:
    //   GB/s = bytes_moved / 1e9 / (time_in_seconds)
    //
    // Note: we mix binary for bytes (sizeof) and decimal 1e9, which is common
    // in practice; for more pedantic reporting you could print GiB/s as well.
    // -------------------------------------------------------------------------
    const double bytes_moved = 3.0 * static_cast<double>(bytes_per_vec);
    const double GB_moved    = bytes_moved / 1e9; // decimal GB

    const double bw_base = GB_moved / (ms_baseline / 1e3); // ms → s
    const double bw_ovl  = GB_moved / (ms_overlap  / 1e3);

    printf("Results:\n");
    printf("  Baseline (1 stream):  %8.3f ms,  %.3f GB moved → %.2f GB/s, checksum = %.6e\n",
           ms_baseline, GB_moved, bw_base, checksum_base);
    printf("  Overlap  (%d streams): %8.3f ms,  %.3f GB moved → %.2f GB/s, checksum = %.6e\n",
           N_STREAMS, ms_overlap, GB_moved, bw_ovl, checksum_ovl);
    printf("  Speedup: %.2fx\n", ms_baseline / ms_overlap);

    // -------------------------------------------------------------------------
    // 9. Cleanup
    // -------------------------------------------------------------------------
    for (int s = 0; s < N_STREAMS; ++s) {
        cudaStreamDestroy(streams[s]);
    }
    cudaEventDestroy(start_all);
    cudaEventDestroy(stop_all);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFreeHost(h_z);

    return 0;
}

