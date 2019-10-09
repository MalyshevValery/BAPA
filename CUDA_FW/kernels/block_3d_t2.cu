//
// Created by malyshev on 9/9/19.
//

#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "../headers/dev_array.h"
#include <chrono>
#include "../headers/block_3d_t2.h"
#include "../../../../../../../usr/include/c++/6/chrono"
#include "../../../../../../../usr/include/c++/6/bits/algorithmfwd.h"
#include <stdlib.h>

using namespace std;

__global__ void I_block_3d_t2_kernel(int n, int r2, int k_start, int k_end, int *matrix) {
    int ROW = threadIdx.y * r2 + k_start;
    int COL = threadIdx.x * r2 + k_start;

    for (int k = k_start; k < k_end; k++) {
        for (int row = ROW; row < ROW + r2; row++)
            for (int col = COL; col < COL + r2; col++)
                matrix[row * n + col] = min(matrix[row * n + col], matrix[row * n + k] + matrix[k * n + col]);
        __syncthreads();
    }
    return;
}

__global__ void SD_block_3d_t2_kernel(int n, int nk, int r, int r2, int *matrix) {
    int tt = blockIdx.x / 2;
    tt = tt + min(1, ((tt + 1) / (nk + 1)));

    int block_row = (blockIdx.x % 2) * nk + (1 - blockIdx.x % 2) * tt;
    int block_col = (1 - blockIdx.x % 2) * nk + (blockIdx.x % 2) * tt;

    int delta_r = block_row * r;
    int delta_c = block_col * r;
    int ROW = (threadIdx.y + block_row * r) * r2;
    int COL = (threadIdx.x + block_col * r) * r2;

    int k_start = nk * R;

    __shared__ int block[R * R];
    __shared__ int central_block[R * R];
    int* row_block = 0;
    int* col_block = 0;
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        for (int i = 0; i < R; i++)
            for (int j = 0; j < R; j++) {
                block[i * R + j] = matrix[k_start * (n + 1) + i * n + j];
                block[i * R + j] = matrix[(delta_r + i) * n + delta_c + j];
            }
    }
    if (blockIdx.x % 2) {
        row_block = &central_block[0];
        col_block = &block[0];
    } else {
        row_block = &block[0];
        col_block = &central_block[0];
    }
    __syncthreads();


    for (int k = 0; k < R; k++) {
        for (int row = ROW; row < ROW + r2; row++)
            for (int col = COL; col < COL + r2; col++)
                block[row * n + col] = min(block[row * n + col], row_block[row * n + k] + col_block[k * n + col]);
        __syncthreads();
    }

    if (threadIdx.y == 0 && threadIdx.x == 0) {
        for (int i = 0; i < R; i++)
            for (int j = 0; j < R; j++)
                matrix[(delta_r + i) * n + delta_c + j] = block[i * R + j];
    }
}

__global__ void DD_block_3d_t2_kernel(int n, int nk, int r, int r2, int *matrix) {
    int block_row = blockIdx.y + min(1, ((blockIdx.y + 1) / (nk + 1)));
    int block_col = blockIdx.x + min(1, ((blockIdx.x + 1) / (nk + 1)));

    int ROW = (threadIdx.y + block_row * r) * r2;
    int COL = (threadIdx.x + block_col * r) * r2;

    int k_start = nk * r * r2;
    int k_end = k_start + r * r2;

    for (int k = k_start; k < k_end; k++) {
        for (int row = ROW; row < ROW + r2; row++)
            for (int col = COL; col < COL + r2; col++)
                matrix[row * n + col] = min(matrix[row * n + col], matrix[row * n + k] + matrix[k * n + col]);
    }
}


void GPU_block_3d_t2_fu(int n, int r, int *matrix) {
    // declare the number of blocks per grid and the number of threads per block
    int nn = n / R;
    int r2 = R / r;
    dim3 threadsPerBlock(r, r);
    dim3 I_blockPerGrid(1, 1);
    dim3 SD_blockPerGrid(2 * (nn - 1));
    dim3 DD_blockPerGrid((nn - 1), (nn - 1));

    //copy data to gpu
    dev_array<int> d_matrix(n * n);
    d_matrix.set(matrix, n * n);

    auto start_hr = std::chrono::high_resolution_clock::now();
    for (int nk = 0; nk < nn; nk++) {
        I_block_3d_t2_kernel << < I_blockPerGrid, threadsPerBlock >> >
                                                  (n, r2, nk * R, (nk + 1) * R, d_matrix.getData());
        SD_block_3d_t2_kernel << < SD_blockPerGrid, threadsPerBlock >> > (n, nk, r, r2, d_matrix.getData());
        DD_block_3d_t2_kernel << < DD_blockPerGrid, threadsPerBlock >> > (n, nk, r, r2, d_matrix.getData());
    }
    auto end_hr = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_hr - start_hr;
    cout << fixed << diff.count() * 1000 << "(ms)" << endl;

    d_matrix.get(matrix, n * n);
}