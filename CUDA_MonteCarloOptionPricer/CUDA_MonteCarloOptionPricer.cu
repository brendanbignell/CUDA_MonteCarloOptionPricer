#include "CUDA_MonteCarloOptionPricer.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

using namespace std;

// Kernel to initialize random states
__global__ void init_random_states(curandState* states, int seed, int num_simulations) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_simulations) return;
    curand_init(seed, idx, 0, &states[idx]);
}

//
// CUDA kernel for generating random asset price paths and calculating payoff for a batch of options
//
// S0: Initial asset price
// K: Strike price
// r: Risk-free rate
// sigma: Volatility
// T: Time to maturity
// num_steps: Number of time steps
// num_simulations: Number of Monte Carlo simulations
// num_options: Number of options in the batch
// d_results: Device memory for option payoffs
//
__global__ void monte_carlo_kernel(curandState* states, float* d_option_prices, float* S0, float* K, 
    float* r, float* sigma, float* T, int num_steps, int num_simulations, int num_options) {
    
    extern __shared__ float shared_payoff[];  // Shared memory for storing partial sums of payoffs
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_simulations * num_options) return;

    int option_idx = idx / num_simulations;
    int sim_idx = idx % num_simulations;

    curandState state = states[sim_idx];  // Load random state for this simulation
    float dt = T[option_idx] / num_steps;  // Time step for this option
    float S = S0[option_idx];              // Initial price for this option

    // Simulate the price path
    for (int step = 0; step < num_steps; ++step) {
        float Z = curand_normal(&state);  // Generate a standard normal random number
        S = S * expf((r[option_idx] - 0.5f * sigma[option_idx] * sigma[option_idx]) * dt + sigma[option_idx] * sqrtf(dt) * Z);
    }

    // Calculate the payoff for a European call option
    //d_results[option_idx * num_simulations + sim_idx] = fmaxf(S - K[option_idx], 0.0f);  // Call option payoff
	float payoff = fmaxf(S - K[option_idx], 0.0f);  // Call option payoff

    // Store payoff in shared memory (one per thread)
    shared_payoff[sim_idx] = payoff;
    __syncthreads();  // Ensure all threads have written their payoffs

    // Use reduction to sum up payoffs within the block (for the current option)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (sim_idx < stride) {
            shared_payoff[sim_idx] += shared_payoff[sim_idx + stride];
        }
        __syncthreads();
    }

    // The first thread in the block calculates the final average payoff and discounts it to present value
    if (sim_idx == 0) {
        float average_payoff = shared_payoff[0] / num_simulations;
        d_option_prices[option_idx] = average_payoff * expf(-r[option_idx] * T[option_idx]);  // Discount to present value
    }

    states[sim_idx] = state;  // Save state back for this simulation
}

// Function to run the Monte Carlo simulation on a specific GPU for multiple options in a batch
void monte_carlo_cuda(int gpu_id, int num_simulations, int num_steps, float* S0, float* K, 
    float* r, float* sigma, float* T, int num_options, float* option_prices) {
    
	cout << "Running Monte Carlo simulation on GPU " << gpu_id << " for " << num_options << " options with " << num_simulations << " simulations each." << endl;

    CUDA_CHECK(cudaSetDevice(gpu_id));  // Set the current device to this GPU

    // Allocate memory for results and random states on the device
    float* d_results;
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_results, num_simulations * num_options * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_states, num_simulations * sizeof(curandState)));

    // Allocate device memory for option parameters
    float* d_S0, * d_K, * d_r, * d_sigma, * d_T, * d_option_prices;
    CUDA_CHECK(cudaMalloc(&d_S0, num_options * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, num_options * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, num_options * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sigma, num_options * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T, num_options * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_option_prices, num_options * sizeof(float)));  // One result per option

    // Copy option parameters to device
    CUDA_CHECK(cudaMemcpy(d_S0, S0, num_options * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K, num_options * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, r, num_options * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma, sigma, num_options * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T, T, num_options * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize random states
    int block_size = 256;
    int num_blocks = (num_simulations + block_size - 1) / block_size;
    init_random_states <<<num_blocks, block_size>>> (d_states, 1234, num_simulations);

    // Launch the Monte Carlo kernel for multiple options in a batch
    //num_blocks = (num_simulations * num_options + block_size - 1) / block_size;
    monte_carlo_kernel << <num_options, num_blocks, block_size >> > (d_states, d_option_prices, d_S0, d_K, d_r, d_sigma, d_T, num_steps, num_simulations, num_options);


    //cudaMemcpy(results, d_results, num_simulations * num_options * sizeof(float), cudaMemcpyDeviceToHost);
    //CUDA_CHECK(cudaMemcpy(results, d_results, num_options * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Copy results back to the host
    CUDA_CHECK(cudaMemcpy(option_prices, d_option_prices, num_options * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    //CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_S0));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_sigma));
    CUDA_CHECK(cudaFree(d_T));
	CUDA_CHECK(cudaFree(d_option_prices));
}

int main() {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    cout << "Number of GPUs: " << num_gpus << endl;

    // Option parameters
    int num_options = 100;
    int num_simulations = 1000000;
	int num_steps = 252;  // Number of time steps (business days in a year)
    int batch_size = 10;  // Number of options per batch

    // Allocate memory for option parameters
    float* S0 = (float*)malloc(num_options * sizeof(float));
    float* K = (float*)malloc(num_options * sizeof(float));
    float* r = (float*)malloc(num_options * sizeof(float));
    float* sigma = (float*)malloc(num_options * sizeof(float));
    float* T = (float*)malloc(num_options * sizeof(float));

    // Generate semi-random option parameters
    /*
    for (int i = 0; i < num_options; ++i) {
        S0[i] = 100.0f + (i * 0.001);
        K[i] = 75.0f + (i * 0.001);
        r[i] = 0.05f + (i * 0.0001f);
        sigma[i] = 0.2f + (i * 0.0001f);
        T[i] = 1.0f;
    }
    */
    // Generate semi-random option parameters
    for (int i = 0; i < num_options; ++i) {
        S0[i] = 100.0f;
        K[i] = 100.0f;
        r[i] = 0.05f;
        sigma[i] = 0.01f;
        T[i] = 1.0f;
    }

    // Allocate memory for results
    float* option_prices = (float*)malloc(num_options * sizeof(float));

    // Stream array for asynchronous execution
    cudaStream_t* streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Launch simulations in batches across multiple GPUs
    for (int i = 0; i < num_options; i += batch_size) {
        int gpu_id = (i / batch_size) % num_gpus;
        int batch_end = min(i + batch_size, num_options);
		//cout << "Running batch " << i / batch_size + 1 << " on GPU " << gpu_id << " for options " << i << " to " << batch_end - 1 << endl;
        monte_carlo_cuda(gpu_id, num_simulations, num_steps, &S0[i], &K[i], &r[i], &sigma[i], &T[i], batch_end - i, &option_prices[i]);
    }

    // Print results
	int num_print = 10;
	cout << "\nLast " << num_print << " Option Prices : " << endl;
    for (int i = 0; i < num_options; ++i) {
        if(i > (num_options - num_print))
            std::cout << "Option " << i + 1 << " Price: " << option_prices[i] << std::endl;
    }
	cout << "\nDone pricing " << num_options << " options with " << num_simulations << " simulations each." << endl;

    // Free memory
    for (int i = 0; i < num_gpus; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
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
