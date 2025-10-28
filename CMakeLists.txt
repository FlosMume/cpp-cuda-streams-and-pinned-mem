cmake_minimum_required(VERSION 3.24)
project(streams_pinned_mem LANGUAGES CXX CUDA)

# Require CUDA 17 standard for host and CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures (Ada 89 by default)
# Adjust as needed, e.g., 86;89;90 for wider builds
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# Warnings & optimization
if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wno-unknown-pragmas)
endif()

add_executable(overlap_streams src/overlap_streams.cu)

# Enable separable compilation (not strictly needed here; useful for larger apps)
set_target_properties(overlap_streams PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Default to Release if not set
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
