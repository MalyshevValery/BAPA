#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "headers/dev_array.h"
#include "headers/block_2d.h"
#include "headers/block_3d.h"
#include "headers/block_3d_t2.h"
#include <fstream>
#include <ctime>
#include <cstring>
#include <thread>
#include <math.h>

#define PLACEHOLDER 999
using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2)
        cout << "No input file";
    else if (argc < 3)
        cout << "No output file";
    else if (argc < 4)
        cout << "No type";

    struct timespec start, finish;
    double elapsed;
    cout.precision(3);
    string type = argv[3];
    int r = 1;
    if (argc == 5)
        r = stoi(string(argv[4]));
    ifstream fin(argv[1]);
    int n;
    fin >> n;
    if (n % r != 0) {
        cout << "Not effective block size" << endl;
        return 0;
    }
    //check cuda properties
    cudaDeviceProp* prop = new cudaDeviceProp();
    cudaError_t err = cudaGetDeviceProperties(prop,0);
    if (err == 0){
        cout << "Device: " << prop->name << endl;
        if (prop->maxThreadsPerBlock < r*r){
            cout << "Device can't handle so many threads in block" << endl;
            cout << "Requested block size: " << r*r << endl;
            cout << "Max threads per block: " << prop->maxThreadsPerBlock << endl;

            cout << "Will attempt second level tailing" << endl;
            if (strcmp(type.c_str(), "block_3d_t2") != 0) {
                cout << "2nd level tailing require 3D block algorithm with 2nd level tailing" << endl;
                return 2;
            }
        }
        else
            cout << "Parameters fit the device" << endl;
    } else {
        cout << "CUDA properties error code: " << err << endl;
        return 2;
    }

    cout << "Reading... ";
    clock_gettime(CLOCK_MONOTONIC, &start);
    int *matrix = new int[n * n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fin >> matrix[i * n + j];
            if (matrix[i * n + j] == -1)
                matrix[i * n + j] = PLACEHOLDER;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout << fixed << elapsed << "s" << endl;

    if (strcmp(type.c_str(), "block_2d") == 0) {
        cout << "2D block FU... ";
        GPU_block_2d_fu(n,r,matrix);
    } else if (strcmp(type.c_str(), "block_3d") == 0) {
        cout << "3D block FU... ";
        GPU_block_3d_fu(n, r, matrix);
    }
    else if (strcmp(type.c_str(), "block_3d_t2") == 0) {
        cout << "3D block FU with 2nd level tailing... " << endl;
        cout << "For shared memory compilation parameter is used:" << endl;
        cout << "Block size " << R << endl;

        cout << "Tailing2 size: " << R2 << endl;
        GPU_block_3d_t2_fu(n, matrix);
    }
    else{
        cout << "No such type" << endl;
        return -1;
    }

    ofstream fout(argv[2]);
    cout << "Writing... ";
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fout << matrix[i * n + j] << "\t";
        }
        fout << endl;
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout << fixed << elapsed << "s" << endl;
    return 0;
}