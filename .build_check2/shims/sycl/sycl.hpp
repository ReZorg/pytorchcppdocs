#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace sycl {
class device {};
class context {};
namespace ext::oneapi::experimental {
enum class queue_state { executing = 0, recording = 1 };
}
class queue {
 public:
  bool ext_oneapi_empty() const { return true; }
  void wait_and_throw() const {}
  ext::oneapi::experimental::queue_state ext_oneapi_get_state() const {
    return ext::oneapi::experimental::queue_state::executing;
  }
};
template <typename T> struct Descriptor { using return_type = T; };
namespace info::device {
struct name : Descriptor<std::string> {}; struct device_type : Descriptor<int> {};
struct vendor : Descriptor<std::string> {}; struct driver_version : Descriptor<std::string> {};
struct version : Descriptor<std::string> {}; struct profile : Descriptor<std::string> {};
struct max_compute_units : Descriptor<uint32_t> {}; struct max_work_item_dimensions : Descriptor<uint32_t> {};
struct max_work_group_size : Descriptor<size_t> {}; struct max_num_sub_groups : Descriptor<uint32_t> {};
struct sub_group_sizes : Descriptor<std::vector<size_t>> {}; struct max_clock_frequency : Descriptor<uint32_t> {};
struct max_mem_alloc_size : Descriptor<uint64_t> {}; struct max_parameter_size : Descriptor<size_t> {};
struct global_mem_size : Descriptor<uint64_t> {}; struct local_mem_size : Descriptor<uint64_t> {};
struct global_mem_cache_size : Descriptor<uint64_t> {}; struct global_mem_cache_line_size : Descriptor<uint32_t> {};
struct global_mem_cache_type : Descriptor<int> {}; struct local_mem_type : Descriptor<int> {};
struct mem_base_addr_align : Descriptor<uint32_t> {};
struct preferred_vector_width_char : Descriptor<uint32_t> {}; struct preferred_vector_width_short : Descriptor<uint32_t> {};
struct preferred_vector_width_int : Descriptor<uint32_t> {}; struct preferred_vector_width_long : Descriptor<uint32_t> {};
struct preferred_vector_width_half : Descriptor<uint32_t> {}; struct preferred_vector_width_float : Descriptor<uint32_t> {};
struct preferred_vector_width_double : Descriptor<uint32_t> {};
struct native_vector_width_char : Descriptor<uint32_t> {}; struct native_vector_width_short : Descriptor<uint32_t> {};
struct native_vector_width_int : Descriptor<uint32_t> {}; struct native_vector_width_long : Descriptor<uint32_t> {};
struct native_vector_width_half : Descriptor<uint32_t> {}; struct native_vector_width_float : Descriptor<uint32_t> {};
struct native_vector_width_double : Descriptor<uint32_t> {};
struct address_bits : Descriptor<uint32_t> {};
struct half_fp_config : Descriptor<std::vector<int>> {}; struct single_fp_config : Descriptor<std::vector<int>> {};
struct double_fp_config : Descriptor<std::vector<int>> {};
struct is_available : Descriptor<bool> {}; struct partition_max_sub_devices : Descriptor<uint32_t> {};
struct profiling_timer_resolution : Descriptor<size_t> {};
}  // namespace info::device
namespace info::platform { struct name : Descriptor<std::string> {}; }
namespace ext::intel::info::device {
struct device_id : Descriptor<uint32_t> {}; struct num_slices : Descriptor<uint32_t> {};
struct num_sub_slices_per_slice : Descriptor<uint32_t> {}; struct num_eus_per_sub_slice : Descriptor<uint32_t> {};
struct num_threads_per_eu : Descriptor<uint32_t> {}; struct gpu_eu_simd_width : Descriptor<uint32_t> {};
struct gpu_hw_threads_per_eu : Descriptor<uint32_t> {}; struct max_mem_bandwidth : Descriptor<uint64_t> {};
struct free_memory : Descriptor<uint64_t> {}; struct memory_clock_rate : Descriptor<uint32_t> {};
struct memory_bus_width : Descriptor<uint32_t> {}; struct gpu_eu_count : Descriptor<uint32_t> {};
struct gpu_eu_count_per_subslice : Descriptor<uint32_t> {};
struct uuid : Descriptor<std::array<unsigned char, 16>> {};
}  // namespace ext::intel::info::device
namespace ext::oneapi::experimental::info::device { struct architecture : Descriptor<int> {}; }
}  // namespace sycl
