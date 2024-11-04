void gemm(float* A, int ldA, 
          float* B, int ldB,
          float* C, int ldC,
          int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i * ldC + j] += A[i * ldA + k] * B[k * ldB + j];
            }
        }
    }
}
