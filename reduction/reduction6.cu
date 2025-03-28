#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include <chrono>
#include <vector>
#include <numeric>

#define BLOCK_DIM 128
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
    Total sum from CPU: 8.38861e+06
    Total sum from GPU: 8.38861e+06
    CPU and GPU results match
    CPU time: 31.2611 ms
    GPU time: 0.7824 ms
    Speedup: 39.9554

    DRAM Throughput   55.51%
*/




// Improvement : Unroll Loops in last warp

/*
- As reduction proceeds, # “active” threads decreases
- When s <= 32, we have only one warp left
- Instructions are SIMD synchronous within a warp
- That means when s <= 32:
    - We don’t need to __syncthreads()
- We don’t need “if (tid < s)” because it doesn’t save any
    work
- Let’s unroll the last 6 iterations of the inner loop
*/



// Problem: Still far way from peak bandwidth, likely bottlenect is instruction overhead
// Strategy: next unroll loops

// Interleaved Addressing
#define CHECK_CUDA_CALL(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Warp reduction with template unrolling
template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >=  8) sdata[tid] += sdata[tid + 4];
    if (blockSize >=  4) sdata[tid] += sdata[tid + 2];
    if (blockSize >=  2) sdata[tid] += sdata[tid + 1];
}

// Reduction kernel template
template <unsigned int blockSize>
__global__ void reduceKernel(float* g_idata, float* g_odata, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    
    int sum = 0;
    while (i < N) {
        sum += g_idata[i];
        if (i + blockDim.x < N) {
            sum += g_idata[i + blockDim.x];
        }
        i += gridSize;
    }
    sdata[tid] = sum;
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void launchReduction(float* d_idata, float* d_odata, int N, int threads) {
    dim3 dimBlock(threads);
    dim3 dimGrid((N + threads - 1) / threads);
    size_t smemSize = threads * sizeof(float);
    
    switch (threads) {
        case 1024: reduceKernel<1024><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 512: reduceKernel<512><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 256: reduceKernel<256><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
        case 128: reduceKernel<128><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
        case  64: reduceKernel<64><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
        case  32: reduceKernel<32><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
        case  16: reduceKernel<16><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
        case   8: reduceKernel<8><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
        case   4: reduceKernel<4><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
        case   2: reduceKernel<2><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
        case   1: reduceKernel<1><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, N); break;
    }
    CHECK_CUDA_CALL(cudaGetLastError());
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

    int N = 4096 * 2048;
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
    int gridSize = (N + BLOCK_DIM - 1) / (BLOCK_DIM * 2);
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
    // Use launchReduction instead of direct kernel call
    launchReduction((float*)input_gpu, (float*)partial_sums_gpu, N, BLOCK_DIM);
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