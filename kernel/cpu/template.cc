#include <iostream>

static int M = 128;
static int N = 128;
static int K = 128;

int main(int argc, char* argv[]) {
    auto A = new float[M * K];
    auto B = new float[K * N];
    auto C = new float[M * N];
    
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}