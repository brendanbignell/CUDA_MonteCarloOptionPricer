#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>
#include <fstream>
#include <filesystem>
#include <mutex>

using namespace std;

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
        curand_init(seed + idx, idx, 0, &randStates[idx]);
    }
}

// Kernel to calculate payoffs for barrier options
__global__ void calculatePayoffs(BarrierOption* options, curandState* randStates, float* payoffs, int numPaths, int numOptions) {
    int optionIdx = blockIdx.x;
    int pathIdx = threadIdx.x + blockDim.x * blockIdx.y;

    if (optionIdx < numOptions && pathIdx < numPaths) {
        BarrierOption option = options[optionIdx];
        curandState localState = randStates[pathIdx];

        int numSteps = static_cast<int>(365 * option.maturity); // Adjust number of steps based on maturity
        float dt = option.maturity / numSteps;
        float spot = option.spot;
        float barrier = option.barrier;
        bool barrierActivated = false;
        float payoff = 0.0f;

        // Simulate path
        for (int i = 0; i < numSteps; i++) {
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

void processBatchOnDevice(int device, int batchStartIdx, int batchEndIdx, int numPaths, const std::vector<BarrierOption>& h_options, std::vector<float>& h_averagePayoffs, std::mutex& resultMutex) {
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
    initRandStates << <numBlocks.y, blockSize >> > (d_randStates, 0, numPaths);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate payoffs
    calculatePayoffs << <numBlocks, blockSize >> > (d_options, d_randStates, d_payoffs, numPaths, numOptionsInBatch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate average payoff for each option
    calculateAveragePayoff << <numOptionsInBatch, 1 >> > (d_payoffs, d_averagePayoffs, numPaths, numOptionsInBatch);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<float> batchAveragePayoffs(numOptionsInBatch);
    CUDA_CHECK(cudaMemcpy(batchAveragePayoffs.data(), d_averagePayoffs, numOptionsInBatch * sizeof(float), cudaMemcpyDeviceToHost));

    // Lock and update the shared host results
    std::lock_guard<std::mutex> guard(resultMutex);
    for (int i = 0; i < numOptionsInBatch; i++) {
        h_averagePayoffs[batchStartIdx + i] = batchAveragePayoffs[i];
    }

    // Free memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_payoffs));
    CUDA_CHECK(cudaFree(d_averagePayoffs));
    CUDA_CHECK(cudaFree(d_randStates));
}

int main() {
    int numOptions = 100 * 1000000;
    int numPaths = 100000;
    int batchSize = 20000; // Set batch size to reduce memory usage
    int numDevices;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));

    std::cout << "Creating " << numOptions << " random barrier options" << endl;

    std::vector<BarrierOption> h_options(numOptions);
    // Generate random data for the batch of options within the specified ranges
    for (int i = 0; i < numOptions; i++) {
        h_options[i] = {
            static_cast<float>(rand() % 10001), // strike
            static_cast<float>(rand() % 10001), // barrier
            static_cast<float>(rand() % 3653) / 365.25f, // maturity upto 10 years in days fraction of a year
            static_cast<float>(rand() % 10001), // spot
            static_cast<float>(rand() % 1001) / 10000.0f, // rate (0 to 10%)
            static_cast<float>(rand() % 1001) / 10000.0f, // volatility (0 to 10%)
            static_cast<BarrierType>(rand() % 4),   // barrierType
            (rand() % 2 == 0) ? European : American, // exerciseType
            (rand() % 2 == 0) ? Call : Put          // optionType
        };
    }

    std::vector<float> h_averagePayoffs(numOptions, 0.0f);
    std::mutex resultMutex;

    // Measure the start time
    auto start = std::chrono::high_resolution_clock::now();

    // Spread load across all available GPUs and batch options concurrently with throttling
    std::vector<std::thread> gpuThreads;
    int maxConcurrentBatches = numDevices * 2; // Limit concurrent batches to prevent GPU memory exhaustion
    int currentBatch = 0;

    while (currentBatch * batchSize < numOptions) {
        while (gpuThreads.size() < maxConcurrentBatches && currentBatch * batchSize < numOptions) {
            int batchStartIdx = currentBatch * batchSize;
            int batchEndIdx = std::min(batchStartIdx + batchSize, numOptions);
            int device = currentBatch % numDevices;
            gpuThreads.emplace_back(processBatchOnDevice, device, batchStartIdx, batchEndIdx, numPaths, std::ref(h_options), std::ref(h_averagePayoffs), std::ref(resultMutex));
            currentBatch++;
        }
        // Join threads that have completed their batch processing
        for (auto it = gpuThreads.begin(); it != gpuThreads.end();) {
            if (it->joinable()) {
                it->join();
                it = gpuThreads.erase(it);
            }
            else {
                ++it;
            }
        }
    }

    // Wait for any remaining threads to finish
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
        std::cout << std::setw(10) << h_options[i].maturity;
        std::cout << std::setw(10) << h_options[i].spot;
        std::cout << std::setw(10) << h_options[i].rate;
        std::cout << std::setw(15) << h_options[i].volatility;
        std::cout << std::setw(15) << (h_options[i].exerciseType == European ? "European" : "American");
        std::cout << std::setw(15) << (h_options[i].barrierType == DownIn ? "DownIn" : (h_options[i].barrierType == DownOut ? "DownOut" : (h_options[i].barrierType == UpIn ? "UpIn" : "UpOut")));
        std::cout << std::setw(15) << h_averagePayoffs[i] << std::endl;
    }

    // Output performance summary
    std::cout << "Total Time Elapsed: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Options Priced per Second: " << optionsPerSecond << "  (" << numPaths << " path each)" << std::endl;

	std::cout << "Writing results to CSV file..." << std::endl;

    // Output the data to a CSV file
    const size_t bufsize = 256 * 1024;
    char buf[bufsize];

    string filePath = std::filesystem::temp_directory_path().string().append("barriers.csv");
    std::ofstream csvFile;
    csvFile.rdbuf()->pubsetbuf(buf, bufsize);
	csvFile.open(filePath);
    
    csvFile << "Strike,Barrier,Maturity,Spot,Rate,Volatility,BarrierType,ExerciseType,OptionType,Price\n";
    for (int i = 0; i < numOptions; i++) {
        csvFile << h_options[i].strike << ","
            << h_options[i].barrier << ","
            << h_options[i].maturity << ","
            << h_options[i].spot << ","
            << h_options[i].rate << ","
            << h_options[i].volatility << ","
            << h_options[i].barrierType << ","
            << h_options[i].exerciseType << ","
            << h_options[i].optionType << ","
            << h_averagePayoffs[i] << "\n";
    }
    csvFile.close();

    std::cout << "Done!" << std::endl;

    return 0;
}
