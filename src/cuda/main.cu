#include <iostream>

__global__ void hello_kernel() {
    printf("Hello from CUDA thread %d!\n", threadIdx.x);
}

int main() {
    hello_kernel<<<1, 4>>>();
    cudaDeviceSynchronize();

    std::cout << "mc_cuda started\n";

    return 0;
}
