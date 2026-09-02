cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(build-check)
find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(EXAMPLES
    cuda/guards cuda/streams cuda/utilities
    xpu/streams xpu/utilities
    frontend/end_to_end)
foreach(src ${EXAMPLES})
  string(REPLACE "/" "_" ex ${src})
  add_executable(${ex} ${CMAKE_CURRENT_SOURCE_DIR}/../generated/examples/${src}.cpp)
  # Local pip-wheel validation only: the wheel omits the CMake-generated
  # *_cmake_macros.h headers, so provide empty shims.
  target_compile_options(${ex} PRIVATE -O2 -ffunction-sections -fdata-sections)
  target_link_options(${ex} PRIVATE -Wl,--gc-sections)
  target_include_directories(${ex} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shims ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nvidia/cuda_runtime/include ${CMAKE_CURRENT_SOURCE_DIR}/vendor/nvidia/cuda_nvcc/include)
  target_link_libraries(${ex} "${TORCH_LIBRARIES}")
endforeach()
# The CPU-only wheel ships no libc10_xpu; satisfy the (never executed) XPU
# device-path symbols with weak stubs.
target_sources(xpu_streams PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/xpu_stubs.cpp)
# The CPU-only wheel ships no libc10_cuda or CUDA runtime; satisfy the
# (never executed) CUDA device-path symbols with weak stubs.
foreach(ex cuda_guards cuda_streams cuda_utilities)
  target_sources(${ex} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_stubs.cpp)
endforeach()
