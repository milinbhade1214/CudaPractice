#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include <chrono>
#include <vector>
#include <numeric>

#define BLOCK_DIM 1024
using namespace std;

/*
Global Synchronization across Multiple SMs is difficult to implement in hardware,
Thus only Blcok level synchronization is possible.

GFLOPs/s -> for compute bound kernels
Bandwidth GB/s -> for memory bound kernels


Reduction have very low arithmetic intensity, thus it is memory bound.
1Flop per Element loaded
*/

/*
    Total sum from CPU: 4.1943e+06
    Total sum from GPU: 4.1943e+06
    CPU and GPU results match
    CPU time: 15.5184 ms
    GPU time: 1.01245 ms
    Speedup: 15.3276

    DRAM Throughput   13.40%
*/




// Improvement : 1. Sequential Addressing, memory coalescing
// Problem: Idle Treads, half of the threads are idle on first loop iteration, so wasteful

// Interleaved Addressing
__global__ void reduction(float *input, float *output, int n){
    __shared__ float sdata[BLOCK_DIM];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();

    // Do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        output[blockIdx.x] = sdata[0];
    }
}


void reduction_cpu(float *input, float *output, int n){
    *output = 0.0f;
    float sum = 0.0f;
    for(int i=0; i<n; i++){
        sum += input[i];
    }   
    *output = sum;
}

int main() {

    int N = 2048 * 2048;
    float* input_cpu = new float[N];
    float output_cpu = 0.0f;

    // Fill host input
    for(int i = 0; i < N; i++){
        input_cpu[i] = 1.0f;
    }

    // Host array for partial sums
    // We'll sum these after copying back from the device
    float* partial_sums_cpu = nullptr;

    // Allocate device memory
    float *input_gpu = nullptr;
    float *partial_sums_gpu = nullptr;

    cudaMalloc(&input_gpu, N * sizeof(float));

    // Adjust grid size for the large array
    int gridSize = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    cudaMalloc(&partial_sums_gpu, gridSize * sizeof(float));

    cudaMemcpy(input_gpu, input_cpu, N * sizeof(float), cudaMemcpyHostToDevice);

    // CPU reduction timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    reduction_cpu(input_cpu, &output_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // GPU reduction timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the kernel with multiple blocks
    reduction<<<gridSize, BLOCK_DIM>>>(input_gpu, partial_sums_gpu, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy partial sums back and do final aggregation on CPU
    partial_sums_cpu = new float[gridSize];
    cudaMemcpy(partial_sums_cpu, partial_sums_gpu, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    float total_gpu = 0.0f;
    for(int i = 0; i < gridSize; i++){
        total_gpu += partial_sums_cpu[i];
    }

    std::cout << "Total sum from CPU: " << output_cpu << std::endl;
    std::cout << "Total sum from GPU: " << total_gpu << std::endl;
    // Compare results
    if(output_cpu == total_gpu){
        std::cout << "CPU and GPU results match" << std::endl;
    } else {
        std::cout << "CPU and GPU results do not match" << std::endl;
    }

    std::cout << "CPU time: " << cpu_time_ms << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;


    std::cout << "Speedup: " << cpu_time_ms / gpu_time_ms << std::endl;

    delete[] input_cpu;
    delete[] partial_sums_cpu;
    cudaFree(input_gpu);
    cudaFree(partial_sums_gpu);

    return 0;
}