#include "CUDA_MonteCarloOptionPricer.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

using namespace std;

// Kernel to initialize random states
__global__ void init_random_states(curandState* states, int seed, int num_simulations) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_simulations) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// CUDA kernel for generating random asset price paths and calculating payoff
__global__ void monte_carlo_kernel(curandState* states, float* d_results, float S0, float K, float r, float sigma, float T, int num_steps, int num_simulations) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_simulations) return;

    curandState state = states[idx];  // Load random state
    float dt = T / num_steps;         // Time step
    float S = S0;                     // Initial price

    // Simulate the price path
    for (int step = 0; step < num_steps; ++step) {
        float Z = curand_normal(&state);  // Generate a standard normal random number
        S = S * expf((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * Z);
    }

    d_results[idx] = fmaxf(S - K, 0.0f);  // Call option payoff
    states[idx] = state;  // Save state back
}

// Function to run the Monte Carlo simulation on a specific GPU
void monte_carlo_cuda(int gpu_id, int num_simulations, int num_steps, float S0, float K, float r, float sigma, float T, float* result) {
    cudaSetDevice(gpu_id);  // Set the current device to this GPU

    // Allocate memory for results and random states on the device
    float* d_results;
    curandState* d_states;
    cudaMalloc(&d_results, num_simulations * sizeof(float));
    cudaMalloc(&d_states, num_simulations * sizeof(curandState));

    // Initialize random states
    int block_size = 256;
    int num_blocks = (num_simulations + block_size - 1) / block_size;
    init_random_states << <num_blocks, block_size >> > (d_states, 1234, num_simulations);

    // Launch the Monte Carlo kernel
    monte_carlo_kernel << <num_blocks, block_size >> > (d_states, d_results, S0, K, r, sigma, T, num_steps, num_simulations);

    // Copy results back to the host
    float* h_results = (float*)malloc(num_simulations * sizeof(float));
    cudaMemcpy(h_results, d_results, num_simulations * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the average payoff and discount to present value
    float payoff_sum = 0.0f;
    for (int i = 0; i < num_simulations; ++i) {
        payoff_sum += h_results[i];
    }
    *result = (payoff_sum / num_simulations) * expf(-r * T);

    // Free memory
    cudaFree(d_results);
    cudaFree(d_states);
    free(h_results);
}

int main() {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    cout << "Number of GPUs: " << num_gpus << endl;

    // Option parameters
    int num_options = 10000;
    int num_simulations = 1000000;
    int num_steps = 252;

    // Allocate memory for option parameters
    float* S0 = (float*)malloc(num_options * sizeof(float));
    float* K = (float*)malloc(num_options * sizeof(float));
    float* r = (float*)malloc(num_options * sizeof(float));
    float* sigma = (float*)malloc(num_options * sizeof(float));
    float* T = (float*)malloc(num_options * sizeof(float));

    // Generate random option parameters
    for (int i = 0; i < num_options; ++i) {
        S0[i] = 100.0f + i;
        K[i] = 75.0f + i;
        r[i] = 0.05f + i * 0.001f;
        sigma[i] = 0.2f + i * 0.001f;
        T[i] = 1.0f;
    }

    // Allocate memory for results
    float* option_prices = (float*)malloc(num_options * sizeof(float));

    // Stream array for asynchronous execution
    cudaStream_t* streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    for (int i = 0; i < num_gpus; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }

    // Launch simulations across multiple GPUs in parallel
    for (int i = 0; i < num_options; ++i) {
        int gpu_id = i % num_gpus;
        monte_carlo_cuda(gpu_id, num_simulations, num_steps, S0[i], K[i], r[i], sigma[i], T[i], &option_prices[i]);
    }

    // Print results
    for (int i = 0; i < num_options; ++i) {
        std::cout << "Option " << i + 1 << " Price: " << option_prices[i] << std::endl;
    }

    // Free memory
    for (int i = 0; i < num_gpus; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    free(S0);
    free(K);
    free(r);
    free(sigma);
    free(T);
    free(option_prices);
    free(streams);

    return 0;
}
