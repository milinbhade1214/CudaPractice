#include<cuda_runtime.h>
#include<stdio.h>
#include<iostream>
#include <vector>
#include <numeric>
#include <functional>
#include <chrono>
#include <cuda.h>



using namespace std;

#define WARP_SIZE 32


// Function to measure GPU execution time of a CUDA kernel with parameters
template <typename Func, typename... Args>
float time_kernel(Func kernel_func, Args&&... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel_func(std::forward<Args>(args)...);  // Pass kernel arguments
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

// Function to perform warmup and benchmark runs
template <typename Func, typename... Args>
float benchmark_kernel(Func kernel_func, int warmup_runs, int benchmark_runs, Args&&... args) {
    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        kernel_func(std::forward<Args>(args)...);
    }

    // Benchmark runs
    std::vector<float> times;
    for (int i = 0; i < benchmark_runs; ++i) {
        float time = time_kernel(kernel_func, std::forward<Args>(args)...);
        times.push_back(time);
    }

    // Calculate average time
    float avg_time = std::accumulate(times.begin(), times.end(), 0.0f) / benchmark_runs;
    return avg_time;
}


// Function to measure execution time of a CPU function
template <typename Func, typename... Args>
double time_cpu(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);  // Execute the function
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();  // Return time in milliseconds
}

// Function to perform warmup and benchmark runs
template <typename Func, typename... Args>
double benchmark_cpu(Func func, int warmup_runs, int benchmark_runs, Args&&... args) {
    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        func(std::forward<Args>(args)...);
    }

    // Benchmark runs
    std::vector<double> times;
    for (int i = 0; i < benchmark_runs; ++i) {
        double time = time_cpu(func, std::forward<Args>(args)...);
        times.push_back(time);
    }

    // Calculate average time
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / benchmark_runs;
    return avg_time;
}




__global__ void softmax_cuda(float *input, float* output, int B, int N){
	extern __shared__ float shared[];

	int tid = threadIdx.x;
	int bid = blockIdx.x + blockIdx.y * gridDim.x;
	int offset = bid * N;

	// Step 1: Compute max (each thread takes max of some elements and store in max_val) 
	float max_val = -INFINITY;
	for(int i=tid; i<N; i+= blockDim.x){
		max_val = fmaxf(max_val, input[offset + i]);
	}

	__shared__ float block_max;
	__syncthreads();
	if (tid == 0) block_max = -INFINITY;
	__syncthreads();

	atomicMax((int*)&block_max, __float_as_int(max_val));
	__syncthreads();
	max_val = block_max;

	// Step 2: Compute sum of exponentials
	float sum = 0.0f;
	for (int i = tid; i < N; i += blockDim.x) {
		sum += expf(input[offset + i] - max_val);
	}

	// Block-wide reduction for sum
	__shared__ float block_sum;
	__syncthreads();
	if (tid == 0) block_sum = 0.0f;
	__syncthreads();

	atomicAdd(&block_sum, sum);
	__syncthreads();
	sum = block_sum;

	// Step 3: Compute softmax values
	for (int i = tid; i < N; i += blockDim.x) {
		output[offset + i] = expf(input[offset + i] - max_val) / sum;
	}
    	
}


void softmax(float *x, int B,  int N) {
    for(int b=0; b<B; b++){
	float max = x[b * N];
	for (int i = 1; i < N; i++) {
        	if (x[b * N + i] > max) {
            		max = x[b * N + i];
        	}
    	}
    	float sum = 0.0;
    	for (int i = 0; i < N; i++) {
        	x[b * N + i] = exp(x[b * N + i] - max);
        	sum += x[b * N + i];
    	}
    	for (int i = 0; i < N; i++) {
        	x[b * N + i] /= sum;
    	}
    }
}

void printValue(float * arr, int values){
	for(int i=0; i<values; i++){
		cout << arr[i] << " " << endl;
	}
	cout << "\n";
}

int main(){
	const int B = 256;  // Batch size
	const int N = 1024;  // Row length
	float *x_cpu = (float*)malloc(B * N * sizeof(float));
	float *x_gpu = (float*)malloc(B * N * sizeof(float));
	float *d_input, *d_output;

	// Initialize input vector
	for (int i = 0; i < B * N; i++) {
		x_cpu[i] = (float)rand() / RAND_MAX;  // Random values between 0 and 1
		x_gpu[i] = x_cpu[i];  // Copy to GPU input
	}

	// Allocate device memory
	cudaMalloc((void**)&d_input, B * N * sizeof(float));
	cudaMalloc((void**)&d_output, B * N * sizeof(float));

	// Copy input data to device
	cudaMemcpy(d_input, x_gpu, B * N * sizeof(float), cudaMemcpyHostToDevice);


	printValue(x_cpu, 10);
	int threadsPerBlock = 256;
	int blocksPerGrid = B;  // One block per batch row
	int sharedMemSize = sizeof(float) * 2;  // Two shared floats (max, sum)


	// Kernel launch wrapper function
	 auto softmax_kernel_func = [&](float* d_input, float* d_output, int B, int N) {
		softmax_cuda<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, B, N);
		cudaDeviceSynchronize();  // Ensure kernel execution is complete
	};

	// Run benchmarking
	float avg_time = benchmark_kernel(softmax_kernel_func, 10, 50, d_input, d_output, B, N);
	printf("Average GPU softmax kernel execution time: %f us\n", avg_time * 1000);




	// Copy result back to host
	cudaMemcpy(x_gpu, d_output, B * N * sizeof(float), cudaMemcpyDeviceToHost);

	printValue(x_gpu, 10);

	// Compute softmax on CPU (for one batch as an example)
	softmax(x_cpu, B, N);

	printValue(x_cpu, 10);
	// Compare results (for the first batch as an example)
	float max_diff = 0.0f;
	for(int b=0; b<B; b++){
	for (int i = 0; i < N; i++) {
		float diff = fabsf(x_cpu[b * N + i] - x_gpu[b * N + i]);
		if (diff > max_diff) {
	    		max_diff = diff;
		}
	}
	}

	printf("Maximum difference between CPU and GPU results (first batch): %e\n", max_diff);


	// Benchmark the CPU softmax function
	double avg_time_cpu  = benchmark_cpu(softmax, 10, 50, x_cpu, B, N);
	std::cout << "Average CPU softmax execution time: " << avg_time_cpu * 1000 << " us\n";


	std::cout << "SpeedUP: " << avg_time_cpu / avg_time << std::endl;


	// Clean up
	free(x_cpu);
	free(x_gpu);
	cudaFree(d_input);
	cudaFree(d_output);

	return 0;	

}
