#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

namespace c10::xpu {

#define XPU_STUB __attribute__((weak))

XPU_STUB XPUStream getCurrentXPUStream(DeviceIndex device) {
  throw std::runtime_error("stub");
}
XPU_STUB XPUStream getStreamFromPool(const bool, DeviceIndex) {
  throw std::runtime_error("stub");
}
XPU_STUB XPUStream getStreamFromPool(const int, DeviceIndex) {
  throw std::runtime_error("stub");
}
XPU_STUB void setCurrentXPUStream(XPUStream) {
  throw std::runtime_error("stub");
}
XPU_STUB void syncStreamsOnDevice(DeviceIndex) {
  throw std::runtime_error("stub");
}
XPU_STUB int XPUStream::priority() const { throw std::runtime_error("stub"); }
XPU_STUB sycl::queue& XPUStream::queue() const {
  throw std::runtime_error("stub");
}
XPU_STUB c10::DeviceIndex device_count() noexcept { return 0; }
XPU_STUB c10::DeviceIndex current_device() { return -1; }
XPU_STUB void set_device(c10::DeviceIndex) {}
XPU_STUB c10::DeviceIndex exchangeDevice(c10::DeviceIndex) { return -1; }
XPU_STUB c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex) { return -1; }
XPU_STUB void device_synchronize() {}
XPU_STUB sycl::device& get_raw_device(DeviceIndex) {
  throw std::runtime_error("stub");
}
XPU_STUB sycl::context& get_device_context() {
  throw std::runtime_error("stub");
}
XPU_STUB void getDeviceProperties(DeviceProp*, DeviceIndex) {}
XPU_STUB void init_device_cache(DeviceIndex) {}
XPU_STUB std::vector<std::optional<std::string>> getDeviceListOfStrings(
    const std::string&) {
  return {};
}
XPU_STUB std::string getDeviceListAsStr() { return ""; }

}  // namespace c10::xpu
