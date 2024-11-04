#include <iostream>
#include "naive.hh"

static int M = 64;
static int N = 64;
static int K = 64;

void increment(float* ptr, int n) {
    float start = 1.;
    for (int i = 0; i < n; ++i) {
        ptr[i] = start++;
    }
}

int main(int argc, char* argv[]) {
    auto A = new float[M * K];
    auto B = new float[K * N];
    auto C = new float[M * N];

    increment(A, M * K);
    increment(B, K * N);
    gemm(A, K, B, N, C, N, M, N, K);
    std::cout << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}