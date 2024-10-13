#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Option types
enum BarrierType { DownIn, DownOut, UpIn, UpOut };
enum ExerciseType { European, American };
enum OptionType { Call, Put };

// Struct to hold barrier option properties
struct BarrierOption {
    float strike;
    float barrier;
    float maturity;
    float spot;
    float rate;
    float volatility;
    BarrierType barrierType;
    ExerciseType exerciseType;
    OptionType optionType;
};

// Kernel to initialize random states for Monte Carlo
__global__ void initRandStates(curandState* randStates, unsigned long seed, int numPaths) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numPaths) {
        curand_init(seed, idx, 0, &randStates[idx]);
    }
}

// Kernel to calculate payoffs for barrier options
__global__ void calculatePayoffs(BarrierOption* options, curandState* randStates, float* payoffs, int numPaths, int numOptions) {
    int optionIdx = blockIdx.x;
    int pathIdx = threadIdx.x + blockDim.x * blockIdx.y;

    if (optionIdx < numOptions && pathIdx < numPaths) {
        BarrierOption option = options[optionIdx];
        curandState localState = randStates[pathIdx];

        float dt = option.maturity / 365.0f;
        float spot = option.spot;
        float barrier = option.barrier;
        bool barrierActivated = false;
        float payoff = 0.0f;

        // Simulate path
        for (int i = 0; i < 365; i++) {
            float gauss_bm = curand_normal(&localState);
            spot *= exp((option.rate - 0.5f * option.volatility * option.volatility) * dt + option.volatility * sqrtf(dt) * gauss_bm);

            if ((option.barrierType == DownIn || option.barrierType == DownOut) && spot <= barrier) {
                barrierActivated = true;
            }
            else if ((option.barrierType == UpIn || option.barrierType == UpOut) && spot >= barrier) {
                barrierActivated = true;
            }

            // Early exercise for American options
            if (option.exerciseType == American) {
                if (option.optionType == Call) {
                    payoff = fmaxf(spot - option.strike, 0.0f);
                }
                else if (option.optionType == Put) {
                    payoff = fmaxf(option.strike - spot, 0.0f);
                }
                // If the current payoff is greater than the previous payoff, exercise early
                if (payoff > 0.0f) {
                    payoffs[optionIdx * numPaths + pathIdx] = payoff * expf(-option.rate * (i * dt));
                    return;
                }
            }
        }

        // Calculate payoff at maturity for European options or if no early exercise occurred for American options
        if (option.optionType == Call) {
            if ((option.barrierType == DownIn || option.barrierType == UpIn) && barrierActivated) {
                payoff = fmaxf(spot - option.strike, 0.0f);
            }
            else if ((option.barrierType == DownOut || option.barrierType == UpOut) && !barrierActivated) {
                payoff = fmaxf(spot - option.strike, 0.0f);
            }
        }
        else if (option.optionType == Put) {
            if ((option.barrierType == DownIn || option.barrierType == UpIn) && barrierActivated) {
                payoff = fmaxf(option.strike - spot, 0.0f);
            }
            else if ((option.barrierType == DownOut || option.barrierType == UpOut) && !barrierActivated) {
                payoff = fmaxf(option.strike - spot, 0.0f);
            }
        }
        payoffs[optionIdx * numPaths + pathIdx] = payoff * expf(-option.rate * option.maturity);
    }
}

// Kernel to calculate average payoff for each option
__global__ void calculateAveragePayoff(float* payoffs, float* averagePayoffs, int numPaths, int numOptions) {
    int optionIdx = blockIdx.x;
    if (optionIdx < numOptions) {
        float sum = 0.0f;
        for (int i = 0; i < numPaths; i++) {
            sum += payoffs[optionIdx * numPaths + i];
        }
        averagePayoffs[optionIdx] = sum / numPaths;
    }
}

void processBatchOnDevice(int device, int batchStartIdx, int batchEndIdx, int numPaths, const std::vector<BarrierOption>& h_options, std::vector<float>& h_averagePayoffs) {
    CUDA_CHECK(cudaSetDevice(device));

    int numOptionsInBatch = batchEndIdx - batchStartIdx;
    BarrierOption* d_options;
    CUDA_CHECK(cudaMalloc(&d_options, numOptionsInBatch * sizeof(BarrierOption)));
    CUDA_CHECK(cudaMemcpy(d_options, &h_options[batchStartIdx], numOptionsInBatch * sizeof(BarrierOption), cudaMemcpyHostToDevice));

    float* d_payoffs;
    CUDA_CHECK(cudaMalloc(&d_payoffs, numOptionsInBatch * numPaths * sizeof(float)));

    float* d_averagePayoffs;
    CUDA_CHECK(cudaMalloc(&d_averagePayoffs, numOptionsInBatch * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_averagePayoffs, 0, numOptionsInBatch * sizeof(float)));

    curandState* d_randStates;
    CUDA_CHECK(cudaMalloc(&d_randStates, numPaths * sizeof(curandState)));

    int blockSize = 256;
    dim3 numBlocks(numOptionsInBatch, (numPaths + blockSize - 1) / blockSize);

    // Initialize random states
    initRandStates << <numBlocks.y, blockSize >> > (d_randStates, time(NULL), numPaths);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate payoffs
    calculatePayoffs << <numBlocks, blockSize >> > (d_options, d_randStates, d_payoffs, numPaths, numOptionsInBatch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate average payoff for each option
    calculateAveragePayoff << <numOptionsInBatch, 1 >> > (d_payoffs, d_averagePayoffs, numPaths, numOptionsInBatch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(&h_averagePayoffs[batchStartIdx], d_averagePayoffs, numOptionsInBatch * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_payoffs));
    CUDA_CHECK(cudaFree(d_averagePayoffs));
    CUDA_CHECK(cudaFree(d_randStates));
}

int main() {
    int numOptions = 1000000;
    int numPaths = 10000;
    int batchSize = 10000; // Set batch size to reduce memory usage
    int numDevices;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));

    std::vector<BarrierOption> h_options(numOptions);
    // Initialize options with some random data (this should be replaced with actual portfolio data)
    for (int i = 0; i < numOptions; i++) {
        h_options[i] = { 100.0f, 90.0f + i % 20, 1.0f, 90.0f + i % 20, 0.01f * (i % 21), 0.01f * (i % 20),
            static_cast<BarrierType>(i % 4), (i % 2 == 0) ? European : American, (i % 2 == 0) ? Call : Put };
    }

    std::vector<float> h_averagePayoffs(numOptions, 0.0f);

    // Measure the start time
    auto start = std::chrono::high_resolution_clock::now();

    // Spread load across all available GPUs and batch options concurrently
    std::vector<std::thread> gpuThreads;
    for (int device = 0; device < numDevices; device++) {
        for (int batchStartIdx = device * batchSize; batchStartIdx < numOptions; batchStartIdx += batchSize * numDevices) {
            int batchEndIdx = std::min(batchStartIdx + batchSize, numOptions);
            gpuThreads.emplace_back(processBatchOnDevice, device, batchStartIdx, batchEndIdx, numPaths, std::ref(h_options), std::ref(h_averagePayoffs));
        }
    }

    // Wait for all threads to finish
    for (auto& thread : gpuThreads) {
        thread.join();
    }

    // Measure the end time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double optionsPerSecond = numOptions / elapsed.count();

    // Output the calculated price for each individual option
    int maxDisplay = numOptions > 20 ? 20 : numOptions;
    std::cout << "\nSummary of Option Parameters and Calculated Prices (First 20):\n";
    std::cout << std::setw(10) << "Option" << std::setw(10) << "Type" << std::setw(10) << "Strike" << std::setw(10) << "Barrier"
        << std::setw(10) << "Spot" << std::setw(10) << "Rate" << std::setw(15) << "Volatility" << std::setw(15) << "Exercise Type"
        << std::setw(15) << "Barrier Type" << std::setw(15) << "Price" << std::endl;
    for (int i = 0; i < maxDisplay; i++) {
        std::cout << std::setw(10) << i + 1;
        std::cout << std::setw(10) << (h_options[i].optionType == Call ? "Call" : "Put");
        std::cout << std::setw(10) << h_options[i].strike;
        std::cout << std::setw(10) << h_options[i].barrier;
        std::cout << std::setw(10) << h_options[i].spot;
        std::cout << std::setw(10) << h_options[i].rate;
        std::cout << std::setw(15) << h_options[i].volatility;
        std::cout << std::setw(15) << (h_options[i].exerciseType == European ? "European" : "American");
        std::cout << std::setw(15) << (h_options[i].barrierType == DownIn ? "DownIn" : (h_options[i].barrierType == DownOut ? "DownOut" : (h_options[i].barrierType == UpIn ? "UpIn" : "UpOut")));
        std::cout << std::setw(15) << h_averagePayoffs[i] << std::endl;
    }

    // Output performance summary
    std::cout << numOptions << " priced in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Options per Second: " << optionsPerSecond << "  (" << numPaths << " path each)" << std::endl;

    return 0;
}