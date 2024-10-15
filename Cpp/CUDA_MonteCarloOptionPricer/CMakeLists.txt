# CMakeList.txt : CMake project for CUDA_MonteCarloOptionPricer, include source and define
# project specific logic here.
#

find_package(CUDAToolkit)

# Add source to this project's executable.
add_executable (CUDA_MonteCarloOptionPricer "CUDA_MonteCarloOptionPricer.cu" "CUDA_MonteCarloOptionPricer.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CUDA_MonteCarloOptionPricer PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
