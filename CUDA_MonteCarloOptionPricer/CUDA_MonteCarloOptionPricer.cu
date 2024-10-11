#include "CUDA_MonteCarloOptionPricer.h"

#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include <device_launch_parameters.h>
#include <curand_kernel.h>  // CUDA Random number generator


using namespace std;

// CUDA kernel for generating random asset price paths and calculating payoff
__global__ void monte_carlo_kernel(float* d_results, float S0, float K, float r, float sigma, float T, int num_steps, int num_simulations) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= num_simulations) return;  // Make sure we don't go out of bounds

    // Initialize the random number generator
    curandState state;
    curand_init(1234, idx, 0, &state);  // Seed, sequence number, and offset

    float dt = T / num_steps;  // Time step
    float S = S0;  // Initial price

    // Simulate the price path
    for (int step = 0; step < num_steps; ++step) {
        float Z = curand_normal(&state);  // Generate a standard normal random number
        S = S * expf((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * Z);
    }

    // Calculate the payoff for a European call option
    d_results[idx] = fmaxf(S - K, 0.0f);  // Call option payoff
}

// Host function to set up and run the Monte Carlo simulation on the GPU
float monte_carlo_cuda(int num_simulations, int num_steps, float S0, float K, float r, float sigma, float T) {
    // Allocate memory for results on the device
    float* d_results;
    cudaMalloc(&d_results, num_simulations * sizeof(float));

    // Define the number of threads and blocks
    int block_size = 256;
    int num_blocks = (num_simulations + block_size - 1) / block_size;

    // Launch the CUDA kernel
    monte_carlo_kernel << <num_blocks, block_size >> > (d_results, S0, K, r, sigma, T, num_steps, num_simulations);

    // Copy results back to host
    float* h_results = (float*)malloc(num_simulations * sizeof(float));
    cudaMemcpy(h_results, d_results, num_simulations * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate the average payoff and discount to present value
    float payoff_sum = 0.0f;
    for (int i = 0; i < num_simulations; ++i) {
        payoff_sum += h_results[i];
    }

    float option_price = (payoff_sum / num_simulations) * expf(-r * T);

    // Free memory
    cudaFree(d_results);
    free(h_results);

    return option_price;
}

int main() {
    // Option parameters
    int num_options = 2000;  // Number of options in the portfolio
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

	cout << "Number of GPUs: " << num_gpus << endl;

    // Allocate memory for option parameters on the host
    float* S0 = (float*)malloc(num_options * sizeof(float));
    float* K = (float*)malloc(num_options * sizeof(float));
    float* r = (float*)malloc(num_options * sizeof(float));
    float* sigma = (float*)malloc(num_options * sizeof(float));
    float* T = (float*)malloc(num_options * sizeof(float));

    // Generate random option parameters
    for (int i = 0; i < num_options; ++i) {
        S0[i] = 100.0f + i;  // Initial stock price
        K[i] = 75.0f + i;   // Strike price
        r[i] = 0.05f + i * 0.001f;    // Risk-free rate
        sigma[i] = 0.2f + i * 0.001f; // Volatility
        T[i] = 1.0f;     // Time to maturity
    }

    int num_simulations = 1000000;  // Number of price paths
    int num_steps = 252;  // Number of time steps (daily steps for 1 year)

    // Allocate memory for results on the host
    float* option_prices = (float*)malloc(num_options * sizeof(float));

    // Run the Monte Carlo simulation for each option in the portfolio
    for (int i = 0; i < num_options; ++i) {
        int gpu_id = i % num_gpus;  // Assign each option to a different GPU
        cudaSetDevice(gpu_id);
        option_prices[i] = monte_carlo_cuda(num_simulations, num_steps, S0[i], K[i], r[i], sigma[i], T[i]);
    }

    // Print the results
    for (int i = 0; i < num_options; ++i) {
        std::cout << "Option " << i + 1 << " Price: " << option_prices[i] << std::endl;
    }

    // Free memory
    free(S0);
    free(K);
    free(r);
    free(sigma);
    free(T);
    free(option_prices);

    return 0;
}
