cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pytorch_cppdocs_generated CXX)

# Locate LibTorch. If it is not installed, configure in skeleton mode:
# targets are still defined (for IDE integration and structure checks)
# but nothing is compiled until Torch is available.
find_package(Torch QUIET)
if(NOT Torch_FOUND)
  message(WARNING "LibTorch not found - defining skeleton targets only. "
          "Re-run with -DCMAKE_PREFIX_PATH=/path/to/libtorch to build.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(CheckLanguage)
check_language(CUDA)

# Non-template reference implementations for the stub interfaces.
file(GLOB GENERATED_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(generated_stubs STATIC ${GENERATED_SOURCES})
target_include_directories(generated_stubs PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(Torch_FOUND)
  target_link_libraries(generated_stubs PUBLIC "${TORCH_LIBRARIES}")
endif()
set_property(TARGET generated_stubs PROPERTY CXX_STANDARD 20)

# One executable per doc-page example (compile-test smoke targets).
add_executable(example_aten_accessors examples/aten/accessors.cpp)
target_link_libraries(example_aten_accessors PRIVATE generated_stubs)
set_property(TARGET example_aten_accessors PROPERTY CXX_STANDARD 20)

add_executable(example_aten_creation examples/aten/creation.cpp)
target_link_libraries(example_aten_creation PRIVATE generated_stubs)
set_property(TARGET example_aten_creation PROPERTY CXX_STANDARD 20)

add_executable(example_aten_indexing examples/aten/indexing.cpp)
target_link_libraries(example_aten_indexing PRIVATE generated_stubs)
set_property(TARGET example_aten_indexing PROPERTY CXX_STANDARD 20)

add_executable(example_aten_tensor examples/aten/tensor.cpp)
target_link_libraries(example_aten_tensor PRIVATE generated_stubs)
set_property(TARGET example_aten_tensor PROPERTY CXX_STANDARD 20)

add_executable(example_autograd_custom_functions examples/autograd/custom_functions.cpp)
target_link_libraries(example_autograd_custom_functions PRIVATE generated_stubs)
set_property(TARGET example_autograd_custom_functions PROPERTY CXX_STANDARD 20)

add_executable(example_autograd_gradient examples/autograd/gradient.cpp)
target_link_libraries(example_autograd_gradient PRIVATE generated_stubs)
set_property(TARGET example_autograd_gradient PROPERTY CXX_STANDARD 20)

add_executable(example_autograd_modes examples/autograd/modes.cpp)
target_link_libraries(example_autograd_modes PRIVATE generated_stubs)
set_property(TARGET example_autograd_modes PROPERTY CXX_STANDARD 20)

add_executable(example_c10_device examples/c10/device.cpp)
target_link_libraries(example_c10_device PRIVATE generated_stubs)
set_property(TARGET example_c10_device PROPERTY CXX_STANDARD 20)

add_executable(example_c10_guards examples/c10/guards.cpp)
target_link_libraries(example_c10_guards PRIVATE generated_stubs)
set_property(TARGET example_c10_guards PROPERTY CXX_STANDARD 20)

add_executable(example_c10_streams examples/c10/streams.cpp)
target_link_libraries(example_c10_streams PRIVATE generated_stubs)
set_property(TARGET example_c10_streams PROPERTY CXX_STANDARD 20)

add_executable(example_c10_types examples/c10/types.cpp)
target_link_libraries(example_c10_types PRIVATE generated_stubs)
set_property(TARGET example_c10_types PROPERTY CXX_STANDARD 20)

add_executable(example_c10_utilities examples/c10/utilities.cpp)
target_link_libraries(example_c10_utilities PRIVATE generated_stubs)
set_property(TARGET example_c10_utilities PROPERTY CXX_STANDARD 20)

add_executable(example_cuda_guards examples/cuda/guards.cpp)
target_link_libraries(example_cuda_guards PRIVATE generated_stubs)
set_property(TARGET example_cuda_guards PROPERTY CXX_STANDARD 20)

add_executable(example_cuda_streams examples/cuda/streams.cpp)
target_link_libraries(example_cuda_streams PRIVATE generated_stubs)
set_property(TARGET example_cuda_streams PROPERTY CXX_STANDARD 20)

add_executable(example_cuda_utilities examples/cuda/utilities.cpp)
target_link_libraries(example_cuda_utilities PRIVATE generated_stubs)
set_property(TARGET example_cuda_utilities PROPERTY CXX_STANDARD 20)

add_executable(example_data_dataloader examples/data/dataloader.cpp)
target_link_libraries(example_data_dataloader PRIVATE generated_stubs)
set_property(TARGET example_data_dataloader PROPERTY CXX_STANDARD 20)

add_executable(example_data_datasets examples/data/datasets.cpp)
target_link_libraries(example_data_datasets PRIVATE generated_stubs)
set_property(TARGET example_data_datasets PROPERTY CXX_STANDARD 20)

add_executable(example_data_samplers examples/data/samplers.cpp)
target_link_libraries(example_data_samplers PRIVATE generated_stubs)
set_property(TARGET example_data_samplers PROPERTY CXX_STANDARD 20)

add_executable(example_data_transforms examples/data/transforms.cpp)
target_link_libraries(example_data_transforms PRIVATE generated_stubs)
set_property(TARGET example_data_transforms PROPERTY CXX_STANDARD 20)

add_executable(example_library_custom_classes examples/library/custom_classes.cpp)
target_link_libraries(example_library_custom_classes PRIVATE generated_stubs)
set_property(TARGET example_library_custom_classes PROPERTY CXX_STANDARD 20)

add_executable(example_library_registration examples/library/registration.cpp)
target_link_libraries(example_library_registration PRIVATE generated_stubs)
set_property(TARGET example_library_registration PROPERTY CXX_STANDARD 20)

add_executable(example_library_versioning examples/library/versioning.cpp)
target_link_libraries(example_library_versioning PRIVATE generated_stubs)
set_property(TARGET example_library_versioning PROPERTY CXX_STANDARD 20)

add_executable(example_nn_activation examples/nn/activation.cpp)
target_link_libraries(example_nn_activation PRIVATE generated_stubs)
set_property(TARGET example_nn_activation PROPERTY CXX_STANDARD 20)

add_executable(example_nn_containers examples/nn/containers.cpp)
target_link_libraries(example_nn_containers PRIVATE generated_stubs)
set_property(TARGET example_nn_containers PROPERTY CXX_STANDARD 20)

add_executable(example_nn_convolution examples/nn/convolution.cpp)
target_link_libraries(example_nn_convolution PRIVATE generated_stubs)
set_property(TARGET example_nn_convolution PROPERTY CXX_STANDARD 20)

add_executable(example_nn_dropout examples/nn/dropout.cpp)
target_link_libraries(example_nn_dropout PRIVATE generated_stubs)
set_property(TARGET example_nn_dropout PROPERTY CXX_STANDARD 20)

add_executable(example_nn_embedding examples/nn/embedding.cpp)
target_link_libraries(example_nn_embedding PRIVATE generated_stubs)
set_property(TARGET example_nn_embedding PROPERTY CXX_STANDARD 20)

add_executable(example_nn_functional examples/nn/functional.cpp)
target_link_libraries(example_nn_functional PRIVATE generated_stubs)
set_property(TARGET example_nn_functional PROPERTY CXX_STANDARD 20)

add_executable(example_nn_linear examples/nn/linear.cpp)
target_link_libraries(example_nn_linear PRIVATE generated_stubs)
set_property(TARGET example_nn_linear PROPERTY CXX_STANDARD 20)

add_executable(example_nn_loss examples/nn/loss.cpp)
target_link_libraries(example_nn_loss PRIVATE generated_stubs)
set_property(TARGET example_nn_loss PROPERTY CXX_STANDARD 20)

add_executable(example_nn_normalization examples/nn/normalization.cpp)
target_link_libraries(example_nn_normalization PRIVATE generated_stubs)
set_property(TARGET example_nn_normalization PROPERTY CXX_STANDARD 20)

add_executable(example_nn_pooling examples/nn/pooling.cpp)
target_link_libraries(example_nn_pooling PRIVATE generated_stubs)
set_property(TARGET example_nn_pooling PROPERTY CXX_STANDARD 20)

add_executable(example_nn_recurrent examples/nn/recurrent.cpp)
target_link_libraries(example_nn_recurrent PRIVATE generated_stubs)
set_property(TARGET example_nn_recurrent PROPERTY CXX_STANDARD 20)

add_executable(example_nn_transformer examples/nn/transformer.cpp)
target_link_libraries(example_nn_transformer PRIVATE generated_stubs)
set_property(TARGET example_nn_transformer PROPERTY CXX_STANDARD 20)

add_executable(example_nn_utilities examples/nn/utilities.cpp)
target_link_libraries(example_nn_utilities PRIVATE generated_stubs)
set_property(TARGET example_nn_utilities PROPERTY CXX_STANDARD 20)

add_executable(example_optim_adaptive examples/optim/adaptive.cpp)
target_link_libraries(example_optim_adaptive PRIVATE generated_stubs)
set_property(TARGET example_optim_adaptive PROPERTY CXX_STANDARD 20)

add_executable(example_optim_gradient_descent examples/optim/gradient_descent.cpp)
target_link_libraries(example_optim_gradient_descent PRIVATE generated_stubs)
set_property(TARGET example_optim_gradient_descent PROPERTY CXX_STANDARD 20)

add_executable(example_optim_schedulers examples/optim/schedulers.cpp)
target_link_libraries(example_optim_schedulers PRIVATE generated_stubs)
set_property(TARGET example_optim_schedulers PROPERTY CXX_STANDARD 20)

add_executable(example_optim_second_order examples/optim/second_order.cpp)
target_link_libraries(example_optim_second_order PRIVATE generated_stubs)
set_property(TARGET example_optim_second_order PROPERTY CXX_STANDARD 20)

add_executable(example_serialize_archives examples/serialize/archives.cpp)
target_link_libraries(example_serialize_archives PRIVATE generated_stubs)
set_property(TARGET example_serialize_archives PROPERTY CXX_STANDARD 20)

add_executable(example_serialize_checkpoints examples/serialize/checkpoints.cpp)
target_link_libraries(example_serialize_checkpoints PRIVATE generated_stubs)
set_property(TARGET example_serialize_checkpoints PROPERTY CXX_STANDARD 20)

add_executable(example_serialize_save_load examples/serialize/save_load.cpp)
target_link_libraries(example_serialize_save_load PRIVATE generated_stubs)
set_property(TARGET example_serialize_save_load PROPERTY CXX_STANDARD 20)

add_executable(example_stable_operators examples/stable/operators.cpp)
target_link_libraries(example_stable_operators PRIVATE generated_stubs)
set_property(TARGET example_stable_operators PROPERTY CXX_STANDARD 20)

add_executable(example_stable_registration examples/stable/registration.cpp)
target_link_libraries(example_stable_registration PRIVATE generated_stubs)
set_property(TARGET example_stable_registration PROPERTY CXX_STANDARD 20)

add_executable(example_stable_utilities examples/stable/utilities.cpp)
target_link_libraries(example_stable_utilities PRIVATE generated_stubs)
set_property(TARGET example_stable_utilities PROPERTY CXX_STANDARD 20)

add_executable(example_xpu_streams examples/xpu/streams.cpp)
target_link_libraries(example_xpu_streams PRIVATE generated_stubs)
set_property(TARGET example_xpu_streams PROPERTY CXX_STANDARD 20)

add_executable(example_xpu_utilities examples/xpu/utilities.cpp)
target_link_libraries(example_xpu_utilities PRIVATE generated_stubs)
set_property(TARGET example_xpu_utilities PROPERTY CXX_STANDARD 20)

if(CMAKE_CUDA_COMPILER AND Torch_FOUND)
  enable_language(CUDA)
  add_executable(example_cuda_kernel examples/cuda/kernel.cu)
  target_link_libraries(example_cuda_kernel PRIVATE generated_stubs)
  set_property(TARGET example_cuda_kernel PROPERTY CUDA_STANDARD 20)
endif()

# Windows: copy LibTorch DLLs next to the binaries.
# See https://github.com/pytorch/pytorch/issues/25457.
if(MSVC AND Torch_FOUND)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  foreach(TARGET_NAME
          example_aten_accessors
          example_aten_creation
          example_aten_indexing
          example_aten_tensor
          example_autograd_custom_functions
          example_autograd_gradient
          example_autograd_modes
          example_c10_device
          example_c10_guards
          example_c10_streams
          example_c10_types
          example_c10_utilities
          example_cuda_guards
          example_cuda_streams
          example_cuda_utilities
          example_data_dataloader
          example_data_datasets
          example_data_samplers
          example_data_transforms
          example_library_custom_classes
          example_library_registration
          example_library_versioning
          example_nn_activation
          example_nn_containers
          example_nn_convolution
          example_nn_dropout
          example_nn_embedding
          example_nn_functional
          example_nn_linear
          example_nn_loss
          example_nn_normalization
          example_nn_pooling
          example_nn_recurrent
          example_nn_transformer
          example_nn_utilities
          example_optim_adaptive
          example_optim_gradient_descent
          example_optim_schedulers
          example_optim_second_order
          example_serialize_archives
          example_serialize_checkpoints
          example_serialize_save_load
          example_stable_operators
          example_stable_registration
          example_stable_utilities
          example_xpu_streams
          example_xpu_utilities
  )
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
      $<TARGET_FILE_DIR:${TARGET_NAME}>)
  endforeach()
endif(MSVC)
