//
// Created by malyshev on 9/9/19.
//

#include <iostream>
#include "cuda_runtime.h"
#include "../headers/dev_array.h"
#include <chrono>
#include "../headers/block_3d_t2.h"
#include <stdlib.h>

using namespace std;

__global__ void I_block_3d_t2_kernel(int n, int r2, int k_start, int k_end, int *matrix) {
    int ROW = threadIdx.y * r2;
    int COL = threadIdx.x * r2;
    int K_SHIFT = k_start * (n + 1);

    __shared__ int block[R * R];
    for (int row = ROW; row < ROW + r2; row++)
        for (int col = COL; col < COL + r2; col++)
            block[row * R + col] = matrix[row * n + col + K_SHIFT];
    __syncthreads();

    for (int k = 0; k < k_end - k_start; k++) {
        for (int row = ROW; row < ROW + r2; row++)
            for (int col = COL; col < COL + r2; col++)
                block[row * R + col] = min(block[row * R + col], block[row * R + k] + block[k * R + col]);
        __syncthreads();
    }

    for (int row = ROW; row < ROW + r2; row++)
        for (int col = COL; col < COL + r2; col++)
            matrix[row * n + col + K_SHIFT] = block[row * R + col];
    return;
}

__global__ void SD_block_3d_t2_kernel(int n, int nk, int r, int *matrix) {
    int tt = blockIdx.x / 2; // index
    bool row_col = blockIdx.x % 2;
    tt = tt + min(1, ((tt + 1) / (nk + 1))); // skip central block
    int block_row, block_col;
    int local_row_inc, central_row_inc;
    int local_col_inc, central_col_inc;
    if (row_col) {
        block_row = nk;
        block_col = tt;
        central_row_inc = R;
        central_col_inc = 0;
        local_col_inc = 1;
        local_row_inc = 0;
    } else {
        block_row = tt;
        block_col = nk;
        local_row_inc = R;
        local_col_inc = 0;
        central_col_inc = 1;
        central_row_inc = 0;
    }

    int ROW = threadIdx.y * R2;
    int COL = threadIdx.x * R2;
    int K_SHIFT = (nk * R) * (n + 1);
    int BLOCK_SHIFT = block_row * R * n + block_col * R;

    __shared__ int local[R * R];
    __shared__ int central[R * R];
    for (int row = ROW; row < ROW + R2; row++)
        for (int col = COL; col < COL + R2; col++) {
            central[row * R + col] = matrix[row * n + col + K_SHIFT];
            local[row * R + col] = matrix[row * n + col + BLOCK_SHIFT];
        }
    __syncthreads();

    int central_idx, local_idx;
    for (int k = 0; k < R; k++) {
        local_idx = ROW * R + k;
        central_idx = ROW * R + k;
        for (int row = ROW; row < ROW + R2; row++) {
            central_idx = (1 - row_col) * (k * R + COL) + row_col * central_idx;
            local_idx = (row_col) *(k * R + COL) + (1 - row_col) * local_idx;
            for (int col = COL; col < COL + R2; col++) {
                local[row * R + col] = min(local[row * R + col], local[local_idx] + central[central_idx]);
                central_idx += central_col_inc;
                local_idx += local_col_inc;
            }
            local_idx += local_row_inc;
            central_idx += central_row_inc;
        }
        __syncthreads();
    }

    for (int row = ROW; row < ROW + R2; row++)
        for (int col = COL; col < COL + R2; col++)
            matrix[row * n + col + BLOCK_SHIFT] = local[row * R + col];
}

__global__ void DD_block_3d_t2_kernel(int n, int nk, int *matrix) {
    int block_row = blockIdx.y + min(1, ((blockIdx.y + 1) / (nk + 1)));
    int block_col = blockIdx.x + min(1, ((blockIdx.x + 1) / (nk + 1)));

    int ROW_START = block_row * R;
    int COL_START = block_col * R;
    int K_START = nk * R;

    int ROW_SHARED = threadIdx.y * R2;
    int COL_SHARED = threadIdx.x * R2;

    int ROW = ROW_SHARED + ROW_START;
    int COL = COL_SHARED + COL_START;

    int result[R2 * R2];
    __shared__ int col_block[R * R];
    __shared__ int row_block[R * R];

    int t2_counter = 0;
    for (int row = ROW_SHARED; row < ROW_SHARED + R2; row++) {
        for (int col = COL_SHARED; col < COL_SHARED + R2; col++) {
            result[t2_counter++] = matrix[(row + ROW_START) * n + col + COL_START];
            col_block[row * R + col] = matrix[(row + ROW_START) * n + col + K_START];
            row_block[row * R + col] = matrix[(row + K_START) * n + col + COL_START];
        }
    }
    __syncthreads();

    for (int k = 0; k < R; k++) {
        t2_counter = 0;
        for (int row = ROW_SHARED; row < ROW_SHARED + R2; row++)
            for (int col = COL_SHARED; col < COL_SHARED + R2; col++)
                result[t2_counter++] = min(result[t2_counter], col_block[row * R + k] + row_block[k * R + col]);
    }

    t2_counter = 0;
    for (int row = ROW; row < ROW + R2; row++)
        for (int col = COL; col < COL + R2; col++)
            matrix[row * n + col] = result[t2_counter++];
}


void GPU_block_3d_t2_fu(int n, int *matrix) {
    // declare the number of blocks per grid and the number of threads per block
    int nn = n / R;
    int r = R / R2;
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
                                                  (n, R2, nk * R, (nk + 1) * R, d_matrix.getData());
        SD_block_3d_t2_kernel << < SD_blockPerGrid, threadsPerBlock >> > (n, nk, r, d_matrix.getData());
        DD_block_3d_t2_kernel << < DD_blockPerGrid, threadsPerBlock >> > (n, nk, d_matrix.getData());
    }
    auto end_hr = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_hr - start_hr;
    cout << fixed << diff.count() * 1000 << "(ms)" << endl;

    d_matrix.get(matrix, n * n);
}