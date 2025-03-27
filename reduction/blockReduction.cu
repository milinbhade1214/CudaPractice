#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include <chrono>
#include <vector>
#include <numeric>

#define BLOCK_DIM 1024
#include <random>

// Initialize random number generator
std::random_device rd;  // Seed
std::mt19937 gen(rd()); // Mersenne Twister PRNG
std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Range [0,1]


using namespace std;

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



__global__ void simpleSumReduction_minimizeGlobalMemAcc(float* input, float *output, int N) {
	__shared__ float input_s[BLOCK_DIM];   
       
	unsigned int i = threadIdx.x;
	input_s[i] = input[i] + input[i + BLOCK_DIM];
    for(unsigned int stride=blockDim.x/2 ; stride >= 1; stride /= 2) {
        	__syncthreads();
	    if (threadIdx.x < stride) {
            input_s[i] += input_s[i + stride];
        }
    }
    if(threadIdx.x == 0) {
        *output = input_s[0];
    }
}






__global__ void simpleSumReduction_minimizeControlDiv(float* input, float *output, int N) {
    unsigned int i = threadIdx.x;
    for(unsigned int stride=blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        *output = input[0];
    }
}


// Fixed GPU Kernel
__global__ void simpleSumReduction_gpu(float* input, float *output, int N) {
    //printf("BlockDim: %d, BlockIdx: %d, ThreadIdx: %d\n", blockDim.x, blockIdx.x, threadIdx.x);
    unsigned int i = 2 * threadIdx.x;
    for(unsigned int stride=1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        *output = input[0];
    }
}

// CPU Sum Reduction
void simpleSumReduction_cpu(float* input, float *output, int N) {
    *output = 0;
    for (int i = 0; i < N; i++) {
        *output += input[i];
    }
}

int main() {
    int N = 2048;
    float* input_cpu = new float[N];
    float output_cpu = 0;
    for (int i = 0; i < N; i++) input_cpu[i] = dis(gen);

    float *input_gpu, *output_gpu;
    cudaMalloc(&input_gpu, N * sizeof(float));
    cudaMalloc(&output_gpu, sizeof(float));

    cudaMemcpy(input_gpu, input_cpu, N * sizeof(float), cudaMemcpyHostToDevice);

    float cpu_time = benchmark_cpu(simpleSumReduction_cpu, 5, 15, input_cpu, &output_cpu, N);
    cout << "CPU Execution time: " << cpu_time << " ms" << endl;
    cout << "Output from CPU: " << output_cpu << endl;

    auto kernel = [input_cpu](float *d_input, float *d_output, int N) { 
        cudaMemcpy(d_input, input_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, sizeof(float));  // Reset output
        simpleSumReduction_gpu<<<1, 1024>>>(d_input, d_output, N); 
        
    };
    float gpu_time = benchmark_kernel(kernel, 5, 15, input_gpu, output_gpu, N);
    
    cudaMemcpy(&output_cpu, output_gpu, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "GPU Execution time: " << gpu_time << " ms" << endl;
    cout << "Output from GPU: " << output_cpu << endl;


    std::cout << "\n\nKernel with minimized control divergence" << std::endl;


	auto kernel1 = [input_cpu](float *d_input, float *d_output, int N) {
        cudaMemcpy(d_input, input_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, sizeof(float));  // Reset output
        simpleSumReduction_minimizeControlDiv<<<1, 1024>>>(d_input, d_output, N);

    };
    float gpu_time1 = benchmark_kernel(kernel1, 5, 15, input_gpu, output_gpu, N);

    cudaMemcpy(&output_cpu, output_gpu, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "GPU Execution time: " << gpu_time1 << " ms" << endl;
    cout << "Output from GPU: " << output_cpu << endl;



	std::cout << "\n\nKernel with reduced Global Mem Access" << std::endl;


        auto kernel2 = [input_cpu](float *d_input, float *d_output, int N) {
        cudaMemcpy(d_input, input_cpu, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, sizeof(float));  // Reset output
        simpleSumReduction_minimizeGlobalMemAcc<<<1, 1024>>>(d_input, d_output, N);

    };
    float gpu_time2 = benchmark_kernel(kernel2, 5, 15, input_gpu, output_gpu, N);

    cudaMemcpy(&output_cpu, output_gpu, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "GPU Execution time: " << gpu_time2 << " ms" << endl;
    cout << "Output from GPU: " << output_cpu << endl;

	






    delete[] input_cpu;
    cudaFree(input_gpu);
    cudaFree(output_gpu);
}
