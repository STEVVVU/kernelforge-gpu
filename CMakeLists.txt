cmake_minimum_required(VERSION 3.24)
project(kernelforge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures")

option(KERNELFORGE_FAST_MATH "Compile kernels with --use_fast_math" ON)

add_library(kernelforge STATIC src/rmsnorm.cu)
target_include_directories(kernelforge PUBLIC include)
target_compile_options(kernelforge PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${KERNELFORGE_FAST_MATH}>>:--use_fast_math>
)

add_executable(kernelforge_bench src/benchmark.cu)
target_link_libraries(kernelforge_bench PRIVATE kernelforge)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME rmsnorm_correctness
        COMMAND kernelforge_bench --rows 64 --hidden 1024 --iterations 5
    )
endif()
