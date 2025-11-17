# CUDA Streams & Pinned Memory — Overlap Compute & Transfers

## 🚀 Overview
This project demonstrates **how to overlap CUDA memory transfers and kernel execution** using:
- Multiple CUDA streams  
- Pinned (page-locked) host memory  
- Asynchronous `cudaMemcpyAsync`  
- A simple SAXPY-like compute (`z = a*x + b`)  

The goal is to show how **PCIe transfers**, **kernel compute**, and **host/device synchronization** can run concurrently to maximize GPU utilization.

---

## 📁 Project Structure
```
streams-and-pinned-mem/
│── CMakeLists.txt
│── overlap_streams.cu
│── README.md  ← (this file)
│── scripts/
│    └── check_cuda_streams_status.sh
│── build/ (generated)
```

---

## ✨ Key Concepts Demonstrated
### 1. CUDA Streams
Each stream executes operations **in order**, but different streams can run **in parallel**:
- Independent **compute** and **memcpy** paths  
- Helps hide PCIe transfer latency  
- Enables multi-chunk pipelining  

### 2. Pinned (Page-Locked) Memory
Pinned memory allows:
- True asynchronous DMA transfers  
- Higher PCIe bandwidth  
- Required for overlap with kernel execution  

Allocated using:
```cpp
cudaHostAlloc(&h_x, N*sizeof(float), cudaHostAllocDefault);
```

### 3. Overlapping Execution
The program uses **N streams**, each responsible for a chunk:
```
H2D copy   →   Kernel   →   D2H copy
```
All streams operate concurrently, creating a pipeline.

---

## 📊 Timeline Diagram (Conceptual)

```
Stream 0: [H2D]----[Compute]-------[D2H]
Stream 1:        [H2D]----[Compute]-------[D2H]
Stream 2:               [H2D]----[Compute]-------[D2H]
Stream 3:                      [H2D]----[Compute]-------[D2H]
```

**Result:** PCIe transfers and kernels run **at the same time**, improving throughput.

---

## 🧮 Kernel Explanation
The compute is intentionally simple:
```cpp
z[i] = a * x[i] + b;
```
This allows the demo to focus on **stream behavior**, not algorithm complexity.

---

## 🛠 Build Instructions (Clean & Simple)

### **Prerequisites**
- Linux (WSL2 Ubuntu recommended)
- NVIDIA GPU + driver
- CUDA Toolkit installed system-wide (`/usr/local/cuda`)

### **Build**
```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### **Run**
```bash
./build/overlap_streams
```

---

## ✔ Verification Script
Included under `scripts/check_cuda_streams_status.sh`:

- Detects `nvcc`  
- Detects GPU compute capability  
- Confirms pinned memory support  
- Prints all CUDA runtime library versions  
- Warns if conda CUDA overrides system CUDA  

Run:
```bash
bash scripts/check_cuda_streams_status.sh
```

---

## 🧪 Tips for Success
### Avoid Conda CUDA Unless Needed
System CUDA is almost always safer:
```bash
which nvcc
# should be /usr/local/cuda/bin/nvcc
```

### Always clear hash after PATH changes
```bash
hash -r
```

### Measure Overlap Efficiency
Use:
```bash
nvprof ./build/overlap_streams
```
or Nsight Systems.

---

## 🔗 References
- NVIDIA CUDA Programming Guide  
- “Streams and Concurrency” — official CUDA samples  
- Nsight Systems Profiling Tutorials  

---

## 👤 Author
**Samuel Huang**  
GitHub: **FlosMume**

---

## 📝 License
MIT License
