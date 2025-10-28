# streams-and-pinned-mem

Overlap host↔device memory copies with GPU compute using **CUDA streams** and **pinned (page-locked) host memory**.

## Highlights
- Uses `cudaMallocHost` for pinned host buffers → enables true async H2D/D2H with `cudaMemcpyAsync`.
- Partitions a large vector into chunks and pipelines **H2D copy → kernel → D2H copy** across multiple streams.
- Simple compute kernel with extra FLOPs to make overlap visible.
- Measures timings with CUDA **events** and prints effective bandwidth & speedup vs. single-stream baseline.

## Build & Run (Linux / WSL / Windows + NVCC)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/overlap_streams
```
(Windows PowerShell): `build\Release\overlap_streams.exe`

## Tunables
Use environment variables to tweak problem size and number of streams:
- `N` (default `16777216`, i.e., 2^24 elements)
- `N_STREAMS` (default `4`)
- `FLOP_ITERS` per element (default `256`) increases compute work

Example:
```bash
N=8388608 N_STREAMS=8 FLOP_ITERS=512 ./build/overlap_streams
```

## Files
- `src/overlap_streams.cu` – demo program
- `CMakeLists.txt` – CUDA 12+ project config (targets Ada, SM 89 by default)
- `scripts/check_streams_status.sh` – quick GPU + build status and micro-benchmark helper
