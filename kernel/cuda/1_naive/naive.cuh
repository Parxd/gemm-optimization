__global__ void naiveKernel(int M, int N, int K,
                            float alpha, const float *A, const float *B,
                            float beta, float *C) {
    // compute (x, y) thread position within block
    const u_int8_t x = (blockDim.x * blockIdx.x) + threadIdx.x;
    const u_int8_t y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (x < M && y < N) {
        float accum = 0.0;
        for (int i = 0; i < K; ++i) {
            accum += A[IDX2C(x, i, K)] * B[IDX2C(i, y, N)];
        }
        C[IDX2C(x, y, N)] = alpha * accum + beta * C[IDX2C(x, y, N)];
    }
}