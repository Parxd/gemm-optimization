add_executable(cuda_naive main.cu)
target_include_directories(cuda_naive PRIVATE naive.cuh)
