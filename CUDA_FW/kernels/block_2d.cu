//
// Created by malyshev on 9/9/19.
//

#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "../headers/dev_array.h"
#include "../headers/block_2d.h"
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
    for (int k = 0; k < n; k++) {
        block_2d_fu_kernel <<< blocksPerGrid, threadsPerBlock >>> (n, k, d_matrix.getData());
        cudaDeviceSynchronize();
        /*d_matrix.get(matrix, n*n);
        cudaDeviceSynchronize();
        d_matrix.set(matrix, n*n);
        cudaDeviceSynchronize();*/
    }
    d_matrix.get(matrix, n*n);
}