#define CUB_STDERR

#include "../../test/test_util.h"
#include <cstdio>
#include <cub/block/block_scan.cuh>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

using namespace cub;

int g_timing_iterations = 1000;

struct Data {
    int64_t val;
    int32_t pre;
    __device__ __host__ Data(int64_t v = 1e15, int32_t p = -1)
        : val(v), pre(p) {}
};

__device__ __host__ bool operator<(const Data &lhs, const Data &rhs) {
    return lhs.val < rhs.val;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
__global__ void BlockPrefixMinKernel(Data *d_in, Data *d_out, int N,
                                     clock_t *d_elapsed) {
    typedef BlockScan<Data, BLOCK_THREADS, ALGORITHM> BlockScanT;
    __shared__ typename BlockScanT::TempStorage temp_storage;
    Data data[ITEMS_PER_THREAD];
    int B = BLOCK_THREADS * ITEMS_PER_THREAD;
    int MAXITER = (N + B - 1) / B;
    Data aggregate;
    for (int iter = 0; iter < MAXITER; iter++) {
        for (int item = 0; item < ITEMS_PER_THREAD; item++) {
            auto idx = iter * B + threadIdx.x * ITEMS_PER_THREAD + item;
            data[item] = idx < N ? d_in[idx] : Data();
        }
        if (threadIdx.x == 0) {
            data[0] = Min()(data[0], aggregate);
        }
        __syncthreads();
        BlockScanT(temp_storage).InclusiveScan(data, data, Min(), aggregate);
        __syncthreads();
        for (int item = 0; item < ITEMS_PER_THREAD; item++) {
            auto idx = iter * B + threadIdx.x * ITEMS_PER_THREAD + item;
            if (idx < N)
                d_out[idx] = data[item];
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
__global__ void BlockPrefixSumKernel(int *d_in, int *d_out, int N,
                                     clock_t *d_elapsed) {
    typedef BlockScan<int, BLOCK_THREADS, ALGORITHM> BlockScanT;
    __shared__ typename BlockScanT::TempStorage temp_storage;
    int data[ITEMS_PER_THREAD];
    int B = BLOCK_THREADS * ITEMS_PER_THREAD;
    int MAXITER = (N + B - 1) / B;
    int aggregate = 0;
    for (int iter = 0; iter < MAXITER; iter++) {
        for (int item = 0; item < ITEMS_PER_THREAD; item++) {
            auto idx = iter * B + threadIdx.x * ITEMS_PER_THREAD + item;
            data[item] = idx < N ? d_in[idx] : 0;
        }
        if (threadIdx.x == 0) {
            data[0] += aggregate;
        }
        __syncthreads();
        BlockScanT(temp_storage).InclusiveSum(data, data, aggregate);
        __syncthreads();
        for (int item = 0; item < ITEMS_PER_THREAD; item++) {
            auto idx = iter * B + threadIdx.x * ITEMS_PER_THREAD + item;
            if (idx < N) {
                d_out[idx] = data[item];
            }
        }
    }
}

int Initialize(int *h_in, int *h_reference, int num_items) {
    int inclusive = 0;

    for (int i = 0; i < num_items; ++i) {
        h_in[i] = i % 17;
        inclusive += h_in[i];
        h_reference[i] = inclusive;
        // std::cout << h_in[i] << " ";
    }
    // std::cout << std::endl;

    return inclusive;
}

void Initialize(Data *h_in, Data *h_reference, int num_items) {
    std::vector<int> s;
    std::random_device rd;
    std::mt19937 g(rd());
    for (int i = 0; i < num_items; i++){
        s.push_back(i);
    }
    std::shuffle(s.begin(), s.end(), g);
    for (int i = 0; i < num_items; i++) {
        h_in[i] = Data(s[i], i);
    }
    h_reference[0] = h_in[0];
    for (int i = 1; i < num_items; i++) {
        h_reference[i] = std::min(h_reference[i - 1], h_in[i]);
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
void testMin(int N) {
    Data *h_in = new Data[N];
    Data *h_ref = new Data[N];
    Data *h_gpu = new Data[N];
    Initialize(h_in, h_ref, N);

    float elapsed = 0;

    Data *d_in = nullptr;
    Data *d_out = nullptr;
    cudaMalloc(&d_in, sizeof(Data) * N);
    cudaMalloc(&d_out, sizeof(Data) * N);
    cudaMemcpy(d_in, h_in, sizeof(Data) * N, cudaMemcpyHostToDevice);
    int max_sm_occupancy;
    CubDebugExit(MaxSmOccupancy(
        max_sm_occupancy,
        BlockPrefixMinKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>,
        BLOCK_THREADS));
    GpuTimer timer;
    timer.Start();
    for (int iter = 0; iter < g_timing_iterations; iter++) {
        BlockPrefixMinKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>
            <<<1, BLOCK_THREADS>>>(d_in, d_out, N, nullptr);
    }
    timer.Stop();
    elapsed += timer.ElapsedMillis();
    cudaMemcpy(h_gpu, d_out, sizeof(Data) * N, cudaMemcpyDeviceToHost);
    for (auto i = 0; i < N; i++) {
        // std::cout << h_ref[i].val << " " << h_ref[i].pre << " " << h_gpu[i].val
        //           << " " << h_gpu[i].pre << std::endl;
        if (h_ref[i].val != h_gpu[i].val || h_ref[i].pre != h_gpu[i].pre) {
            printf("RESULT WRONG %d\n", i);
            exit(-1);
        }
    }
    std::cout << "Elasped Per Kernel : " << elapsed / g_timing_iterations
              << " ms" << std::endl;
    std::cout << max_sm_occupancy << " SM occupancy" << std::endl;
    delete h_in;
    delete h_ref;
    delete h_gpu;
    cudaFree(d_in);
    cudaFree(d_out);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
void test(int N) {
    int *h_in = new int[N];
    int *h_ref = new int[N];
    int *h_gpu = new int[N];

    Initialize(h_in, h_ref, N);

    float elapsed = 0;

    int *d_in = nullptr;
    int *d_out = nullptr;
    clock_t *d_elasped = nullptr;
    cudaMalloc(&d_in, sizeof(int) * N);
    cudaMalloc(&d_out, sizeof(int) * N);
    cudaMalloc(&d_elasped, sizeof(clock_t));
    cudaMemcpy(d_in, h_in, sizeof(int) * N, cudaMemcpyHostToDevice);
    int max_sm_occupancy;
    CubDebugExit(MaxSmOccupancy(
        max_sm_occupancy,
        BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>,
        BLOCK_THREADS));
    GpuTimer timer;
    timer.Start();
    for (int iter = 0; iter < g_timing_iterations; iter++) {
        BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>
            <<<1, BLOCK_THREADS>>>(d_in, d_out, N, d_elasped);
    }
    timer.Stop();
    elapsed += timer.ElapsedMillis();
    cudaMemcpy(h_gpu, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for (auto i = 0; i < N; i++) {
        // std::cout << h_ref[i] << " " << h_gpu[i] << std::endl;
        if (h_ref[i] != h_gpu[i]) {
            printf("RESULT WRONG %d\n", i);
            exit(-1);
        }
    }
    std::cout << "Elasped Per Kernel : " << elapsed / g_timing_iterations
              << " ms" << std::endl;
    std::cout << max_sm_occupancy << " SM occupancy" << std::endl;
    delete h_in;
    delete h_ref;
    delete h_gpu;
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_elasped);
}

int main() {
    test<1024, 1, BLOCK_SCAN_RAKING>(100000);
    testMin<1024, 1, BLOCK_SCAN_RAKING>(100000);
}