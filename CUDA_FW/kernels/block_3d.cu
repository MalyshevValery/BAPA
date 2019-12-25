//
// Created by malyshev on 9/9/19.
//

#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "../headers/dev_array.h"
#include "../headers/block_3d.h"
#include <stdlib.h>
#include <chrono>

using namespace std;

__global__ void I_block_3d_kernel(int n, int k_start, int k_end, int *matrix) {
    int ROW = threadIdx.y + k_start;
    int COL = threadIdx.x + k_start;

    for (int k = k_start; k < k_end; k++) {
        matrix[ROW * n + COL] = min(matrix[ROW * n + COL], matrix[ROW * n + k] + matrix[k * n + COL]);
        __syncthreads();
    }
    return;
}

__global__ void SD_block_3d_kernel(int n, int nk, int r, int *matrix) {
    int tt = blockIdx.x / 2;
    tt = tt + min(1,((tt + 1) / (nk + 1)));

    int block_row = (blockIdx.x % 2) * nk + (1 - blockIdx.x % 2) * tt;
    int block_col = (1 - blockIdx.x % 2) * nk + (blockIdx.x % 2) * tt;

    int ROW = threadIdx.y + block_row * r;
    int COL = threadIdx.x + block_col * r;

    int k_start = nk * r;
    int k_end = k_start + r;

    for (int k = k_start; k < k_end; k++) {
        matrix[ROW * n + COL] = min(matrix[ROW * n + COL], matrix[ROW * n + k] + matrix[k * n + COL]);
        __syncthreads();
    }
}

__global__ void DD_block_3d_kernel(int n, int nk, int r, int *matrix) {
    int block_row = blockIdx.y + min(1,((blockIdx.y + 1) / (nk + 1)));
    int block_col = blockIdx.x + min(1,((blockIdx.x + 1) / (nk + 1)));

    int ROW = threadIdx.y + block_row * r;
    int COL = threadIdx.x + block_col * r;

    int k_start = nk * r;
    int k_end = k_start + r;

    for (int k = k_start; k < k_end; k++) {
        matrix[ROW * n + COL] = min(matrix[ROW * n + COL], matrix[ROW * n + k] + matrix[k * n + COL]);
    }
}


void GPU_block_3d_fu(int n, int r, int *matrix) {
    // declare the number of blocks per grid and the number of threads per block
    int nn = n / r;
    dim3 threadsPerBlock(r, r);
    dim3 I_blockPerGrid(1, 1);
    dim3 SD_blockPerGrid(2 * (nn - 1));
    dim3 DD_blockPerGrid((nn - 1),(nn - 1));

    //copy data to gpu
    dev_array<int> d_matrix(n * n);
    d_matrix.set(matrix, n * n);

    auto start_hr = std::chrono::high_resolution_clock::now();
    for (int nk = 0; nk < nn; nk++) {
        I_block_3d_kernel << < I_blockPerGrid, threadsPerBlock >> > (n, nk * r, (nk + 1) * r, d_matrix.getData());
        SD_block_3d_kernel << < SD_blockPerGrid, threadsPerBlock >> > (n, nk, r, d_matrix.getData());
        DD_block_3d_kernel << < DD_blockPerGrid, threadsPerBlock >> > (n, nk, r, d_matrix.getData());
    }
    auto end_hr = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_hr - start_hr;
    cout << fixed << diff.count() * 1000 << "(ms)" << endl;

    d_matrix.get(matrix, n * n);
}