#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include <chrono>
#include <vector>
#include <numeric>

#define BLOCK_DIM 1024
using namespace std;

#include <random>

// Initialize random number generator
std::random_device rd;  // Seed
std::mt19937 gen(rd()); // Mersenne Twister PRNG
std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Range [0,1]


// Function to measure GPU Execution time
template <typename Func, typename... Args>
float time_kernel(Func kernel_func, Args&&... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();  // Ensure previous operations are done
    cudaEventRecord(start);

    kernel_func(std::forward<Args>(args)...);  // Kernel launch

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);  // Wait for kernel completion

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

// Function to benchmark kernel
template <typename Func, typename... Args>
float benchmark_kernel(Func kernel_func, int warmup_runs, int benchmark_runs, Args&&... args) {
    for (int i = 0; i < warmup_runs; ++i) {
        kernel_func(std::forward<Args>(args)...);
        cudaDeviceSynchronize();
    }

    std::vector<float> times;
    for (int i = 0; i < benchmark_runs; ++i) {
        float time = time_kernel(kernel_func, std::forward<Args>(args)...);
        times.push_back(time);
    }

    return std::accumulate(times.begin(), times.end(), 0.0f) / benchmark_runs;
}

// Function to benchmark CPU
template <typename Func, typename... Args>
double benchmark_cpu(Func func, int warmup_runs, int benchmark_runs, Args&&... args) {
    for (int i = 0; i < warmup_runs; ++i) {
        func(std::forward<Args>(args)...);
    }

    std::vector<double> times;
    for (int i = 0; i < benchmark_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();

        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    return std::accumulate(times.begin(), times.end(), 0.0) / benchmark_runs;
}


// Segmented MultiBlock Reduction using Atomic Operation

__global__ void sumReduction(float *input, float* output, long long N){
	__shared__ float input_s[BLOCK_DIM];
	unsigned int segment = 2 * blockDim.x * blockIdx.x;
	unsigned int i = segment + threadIdx.x;
	unsigned int t = threadIdx.x;

	float sum = 0.0f;
	if(i < N) sum += input[i];
	if(i + BLOCK_DIM < N) sum += input[i + BLOCK_DIM];
	input_s[t] = sum;

	for(unsigned int stride = blockDim.x/2; stride >=1; stride/=2){
		__syncthreads();
		if(t<stride){
			input_s[t] += input_s[t+stride];
		}
	}

	// Atomic add is done across blocks, all blocks will dump their input_s[0] value in otuput which is atomically added
	if(t == 0){
		atomicAdd(output, input_s[0]); // Just this changed
	}
}

// CPU Sum Reduction
void simpleSumReduction_cpu(float* input, float *output, long long N) {
    *output = 0;
    double sum = 0.0;
    
    for (long long i = 0; i < N; i++) {
        sum += input[i];
    }
    *output = static_cast<float>(sum);
}



int main() {
    long long N = 8192LL * 8192;
    float* input_cpu = new float[N];
    float output_cpu = 0.0f;
    
    for (int i = 0; i < N; i++) input_cpu[i] = 1.0;

    float *input_gpu, *output_gpu;
    cudaMalloc(&input_gpu, N * sizeof(float));
    cudaMalloc(&output_gpu, sizeof(float));

    cudaMemcpy(input_gpu, input_cpu, N * sizeof(float), cudaMemcpyHostToDevice);

    float cpu_time = benchmark_cpu(simpleSumReduction_cpu, 5, 15, input_cpu, &output_cpu, N);
    cout << "CPU Execution time: " << cpu_time << " ms" << endl;
    cout << "Output from CPU: " << output_cpu << endl;


    long long blocksPerGrid = (N + 1024 - 1)/(1024 * 2);
    auto kernel = [input_cpu, blocksPerGrid](float *d_input, float *d_output, long long  N) { 
        cudaMemcpy(d_input, input_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, sizeof(float));  // Reset output
        sumReduction<<<blocksPerGrid, 1024>>>(d_input, d_output, N); 
        
    };
    float gpu_time = benchmark_kernel(kernel, 5, 15, input_gpu, output_gpu, N);
    
    cudaMemcpy(&output_cpu, output_gpu, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "GPU Execution time: " << gpu_time << " ms" << endl;
    cout << "Output from GPU: " << output_cpu << endl;

	

    delete[] input_cpu;
    cudaFree(input_gpu);
    cudaFree(output_gpu);
}




