// Examples for the "Streams" page (api/c10/streams).
// Covers c10::Stream, the device-agnostic base stream class. Streams are
// obtained from backend-specific APIs; when a CUDA build and device are
// available we use c10::cuda::getCurrentCUDAStream().
#include <torch/torch.h>

#include <c10/core/Stream.h>
#include <iostream>
#include <optional>

#if defined(USE_CUDA) || defined(TORCH_HAS_CUDA) || defined(C10_USING_CUDA)
#include <c10/cuda/CUDAStream.h>
#define HAVE_CUDA_STREAM_API 1
#else
#define HAVE_CUDA_STREAM_API 0
#endif

namespace {

// Exercises the common c10::Stream interface described on the doc page.
void describe_stream(const c10::Stream& stream) {
  c10::Device device = stream.device();
  c10::DeviceType type = stream.device_type();
  std::cout << "stream device: " << device << std::endl;
  std::cout << "stream device_type: " << type << std::endl;
  std::cout << "stream device_index: " << stream.device_index() << std::endl;
  std::cout << "stream id: " << stream.id() << std::endl;
  c10::StreamData3 packed = stream.pack3();
  std::cout << "stream pack3/unpack3 round-trip equal: "
            << (c10::Stream::unpack3(packed.stream_id, packed.device_index,
                                     packed.device_type) == stream)
            << std::endl;
}

}  // namespace

int main() {
#if HAVE_CUDA_STREAM_API
  if (torch::cuda::is_available()) {
    // Streams are typically obtained from backend-specific APIs.
    c10::cuda::CUDAStream cuda_stream = c10::cuda::getCurrentCUDAStream();

    // c10::Stream provides the common, device-agnostic interface.
    describe_stream(cuda_stream);

    // A non-default stream from the backend-specific pool.
    c10::cuda::CUDAStream side_stream = c10::cuda::getStreamFromPool();
    std::cout << "pooled stream id != current stream id: "
              << (side_stream.id() != cuda_stream.id()) << std::endl;

    // Work queued on the side stream via the backend-specific guard.
    {
      c10::cuda::CUDAStreamGuard stream_guard(side_stream);
      torch::Tensor on_side_stream =
          torch::randn({8, 8}, c10::Device(c10::kCUDA, 0));
      side_stream.synchronize();
      std::cout << "side-stream result sum: "
                << on_side_stream.sum().item<float>() << std::endl;
    }
    return 0;
  }
#endif
  std::cout << "CUDA not available; c10::Stream values are produced by "
               "backend-specific APIs at runtime (e.g. "
               "c10::cuda::getCurrentCUDAStream())."
            << std::endl;
  std::cout << "c10::Stream exposes device(), device_type(), "
               "device_index(), id(), and pack3()/unpack3()."
            << std::endl;
  return 0;
}
