#include <iostream>
#include "../../utils.cuh"
#include "naive.cuh"

static int M, N, K;

int main(int argc, char* argv[]) {
    M = std::stoi(argv[1]);
    N = std::stoi(argv[2]);
    K = std::stoi(argv[3]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    float *d_A, *d_B, *d_C;
    auto A = new float[M * K];
    auto B = new float[K * N];
    fill_increment(A, M * K);
    fill_increment(B, K * N);
    
    auto C = new float[M * N];
    auto A_size = sizeof(float) * M * K;
    auto B_size = sizeof(float) * K * N;
    auto C_size = sizeof(float) * M * N;
    cudaMallocAsync((void**)&d_A, A_size, stream);
    cudaMallocAsync((void**)&d_B, B_size, stream);
    cudaMallocAsync((void**)&d_C, C_size, stream);
    cudaMemcpyAsync(d_A, A, A_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B, B_size, cudaMemcpyHostToDevice, stream);
    
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);  // avoid grid & block z-axis for now
    dim3 blockDim(32, 32, 1);
    naiveKernel<<<gridDim, blockDim, 0, stream>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpyAsync(C, d_C, C_size, cudaMemcpyDeviceToHost, stream);
    // cudaStreamSynchronize(stream);
    
    print(C, M, N);
    
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFreeAsync(d_A, stream);
    cudaFreeAsync(d_B, stream);
    cudaFreeAsync(d_C, stream);
    cudaStreamDestroy(stream);
    return 0;
}