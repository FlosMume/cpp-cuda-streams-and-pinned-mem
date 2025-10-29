# 🖼️ cpp-cuda-image-filter

A CUDA C++ project that implements a **2D image convolution** (e.g., a 5×5 blur filter) using **shared memory tiling** for performance optimization.  
It demonstrates how to accelerate image-processing tasks by leveraging **data locality**, **parallelism**, and **on-chip shared memory** on modern NVIDIA GPUs.

---

## 🎯 Learning Objectives

By completing this project, you will learn to:

1. **Apply shared memory**  
   Use `__shared__` memory to cache image tiles and reduce global memory latency.

2. **Understand 2D tiling and halo regions**  
   Manage **tile borders (aprons)** to correctly compute convolutions near image edges.

3. **Implement and test convolution kernels**  
   Build a CUDA kernel that applies a 5×5 normalized box blur filter.

4. **Optimize memory access patterns**  
   Learn **coalesced reads/writes** and how block and grid dimensions affect throughput.

5. **Transfer 2D data between host and device**  
   Allocate and copy large 2D arrays with `cudaMalloc` and `cudaMemcpy`.

6. **Structure CUDA projects professionally**  
   Configure builds using **CMake**, debug via **VS Code**, and maintain clean, reproducible code.

---

## 🧠 Core Topics

| Category | Topics Covered |
|-----------|----------------|
| **CUDA Programming Model** | Thread/block/grid indexing in 2D; mapping `(x, y)` coordinates |
| **Memory Hierarchy** | Global vs. shared vs. constant memory (`__shared__`, `__constant__`) |
| **Image Processing Basics** | 2D convolution, box filter, normalization, border clamping |
| **Tiling Optimization** | Shared memory tiles with halo (apron) handling  |
| **Performance Concepts** | Data reuse, coalesced access, warp divergence avoidance |
| **Project Engineering** | Clean `CMakeLists.txt`, reproducible builds, error-checked CUDA calls |

---

## 🧩 Example Output

```bash
Center value after blur: 0.040000 (expected ~0.04)
Success!
```
A simple impulse test verifies that a single bright pixel spreads correctly into a 5×5 blurred patch.

---

## ⚙️ Build & Run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/image_filter
```

---

## 🧱 Shared Memory Tile Diagram

Below is a conceptual diagram of how tiles and halo regions are loaded into shared memory for a 5×5 convolution kernel.

```
Global Image (Input)

 ┌────────────────────────────────────────────┐
 │         ... previous blocks ...            │
 │ ┌──────────────┬──────────────┬──────────┐ │
 │ │   Halo (+2)  │  TILE (16×16)│  Halo    │ │
 │ │   region     │   region     │  region  │ │
 │ ├──────────────┼──────────────┼──────────┤ │
 │ │              │              │          │ │
 │ │<-- shared memory (TILE+4)² -->         │ │
 │ │  (TILE=16, K=5, R=2)                   │ │
 │ └────────────────────────────────────────┘ │
 │        ... next blocks ...                 │
 └────────────────────────────────────────────┘
```

**Legend:**
- `R = K/2` → filter radius (e.g., 2 for 5×5 kernel)
- Threads in a block load `(TILE + 2R) × (TILE + 2R)` elements into shared memory
- The inner `TILE×TILE` region performs actual convolution output

---

## 🧭 Next Step
This project prepares you for the **`streams-and-pinned-mem`** module —  
where you’ll learn to **overlap data transfers with computation** for higher throughput.

---

## 📄 License
This project is open-sourced under the MIT License.

---

**Author:** Samuel Huang  
**Repo:** [FlosMume/cpp-cuda-image-filter](https://github.com/FlosMume/cpp-cuda-image-filter)
