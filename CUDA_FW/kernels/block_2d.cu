//
// Created by malyshev on 9/9/19.
//

#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "../headers/dev_array.h"
#include "../headers/block_2d.h"
#include <chrono>
#include <stdlib.h>

using namespace std;

__global__ void block_2d_fu_kernel(int n, int k, int* matrix) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    matrix[ROW * n + COL] = min(matrix[ROW * n + COL], matrix[ROW * n + k] + matrix[k * n + COL]);
    return;
}


void GPU_block_2d_fu(int n, int r, int* matrix) {
    // declare the number of blocks per grid and the number of threads per block
    int nn = n / r;
    dim3 threadsPerBlock(r, r);
    dim3 blocksPerGrid(nn, nn);

    //copy data to gpu
    dev_array<int> d_matrix(n*n);
    d_matrix.set(matrix, n*n);

    auto start_hr = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < n; k++) {
        block_2d_fu_kernel <<< blocksPerGrid, threadsPerBlock >>> (n, k, d_matrix.getData());
    }
    auto end_hr = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_hr - start_hr;
    cout << fixed << diff.count() * 1000 << "(ms)" << endl;

    d_matrix.get(matrix, n*n);
}