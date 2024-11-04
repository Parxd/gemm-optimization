#include <iostream>
#include <vector>
#include <omp.h>

static int M = 128;
static int N = 128;
static int K = 128;

int main(int argc, char* argv[]) {
    int size = 500000000;
    std::vector<int> A(size, 1);
    std::vector<int> B(size, 1);
    std::vector<int> C(size, 0);
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }

    // for (int i = 0; i < size; ++i) {
    //     std::cout << C[i] << " ";
    // }
    return 0;
}