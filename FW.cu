#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <iostream>

//const bool DEBUG = true;
//Размер тайла первого уровня
#define R1 64

//Размер тайла второго уровня
#define R2 2

#define INF INT32_MAX / 2

//Деление с округлением вверх
int ceil(const int a, const int b);

//#define Q2 ceil(R1, R2)
#define Q2 (R1 / R2)

//вспомогательные величины для оптимальной загрузки из глобальной памяти
#define S1 ((R2 * R2 > R1) ? R1 : R2 * R2)
#define T1 (R1 / S1)
#define T2 ((S1 < R1) ? R1 : Q2 * Q2)
#define S2 (R1 / T2)

#define INPUT_PATH "CUDA_FW/graph.txt"
#define OUTPUT_PATH "output.txt"
#define OUTPUT_TRUE_PATH "output_true.txt"
//n - vertices, m - edges
int n, m;

__global__ void kernelI(int * deviceGraph, int pitch, int n, int k);
__global__ void kernelSD(int* deviceGraph, int pitch, int n, int k);
__global__ void kernelDD(int* deviceGraph, int pitch, int n, int k);

cudaError_t cudaStatus = cudaSetDevice(0);

int* readGraph() {
	FILE* inputFile = fopen(INPUT_PATH, "r");
	fscanf(inputFile, "%d %d", &n, &m);
	int* graph;
	cudaStatus = cudaMallocHost(&graph, sizeof(int) * n * n);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMallocHost failed!");
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			graph[i * n + j] = (i != j) ? INF : 0;
		}
	}

	for (int i = 0; i < m; ++i) {
		int u, v, d;
		fscanf(inputFile, "%d %d %d", &u, &v, &d);
		graph[u * n + v] = d;
	}

	fclose(inputFile);
	return graph;
}

void writeResult(int32_t * hostGraph) {
	FILE* outputFile = fopen("output.txt", "w");
	fprintf(outputFile, "%d\n", n);
	int k = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			fprintf(outputFile, "%d ", hostGraph[k]);
			++k;
		}
		fprintf(outputFile, "\n");
	}
	fclose(outputFile);
}

int ceil(const int a, const int b) {
	return (a + b - 1) / b;
}

bool check(int* graph) {
	FILE* file = fopen(OUTPUT_TRUE_PATH, "r");
	int n1;
	fscanf(file, "%d", &n1);
	for (int i = 0; i < n1; ++i) {
		for (int j = 0; j < n1; ++j) {
			int e;
			fscanf(file, "%d", &e);
			if (e != graph[i * n + j]) {
				fclose(file);
				return false;
			}
		}
	}
	fclose(file);
	return true;
}

__global__ void wakeGPU() {
	int i = threadIdx.x;
}

int main() {

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//host located graph
	int* hostGraph = readGraph();
	printf("Graph is loaded\n");

	const int Q1 = ceil(n, R1);
	if (Q1 * R1 != n) {
		printf("n should divide by R1\n");
		return 1;
	}

	int* deviceGraph = 0;
	size_t pitch;
	int pitchInt;
	cudaStatus = cudaMallocPitch((void**)& deviceGraph, &pitch, (size_t)(n * sizeof(int32_t)), (size_t)n);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaMallocPitch failed!");
	}
	cudaStatus = cudaMemcpy2D(deviceGraph, pitch, hostGraph, n * sizeof(int32_t), n * sizeof(int32_t), n, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D failed!");
	}
	assert(!(pitch % sizeof(int)));
	pitchInt = pitch / sizeof(int);

	dim3 gridI(1);
	dim3 blockI(Q2, Q2);

	dim3 gridSD(Q1 - 1, 2);
	dim3 blockSD(Q2, Q2);

	dim3 gridDD(Q1 - 1, Q1 - 1);
	dim3 blockDD(Q2, Q2);

	cudaEvent_t stepFinishedEvent;
	cudaEventCreate(&stepFinishedEvent);

	wakeGPU<<<1, 1>>>();

	auto start = std::chrono::high_resolution_clock::now();
	for (int k = 0; k < Q1; ++k) {
		kernelI<<<gridI, blockI>>>(deviceGraph, pitchInt, n, k);
		cudaEventRecord(stepFinishedEvent);
		cudaEventSynchronize(stepFinishedEvent);
		kernelSD<<<gridSD, blockSD>>>(deviceGraph, pitchInt, n, k);
		cudaEventRecord(stepFinishedEvent);
		cudaEventSynchronize(stepFinishedEvent);
		kernelDD<<<gridDD, blockDD>>>(deviceGraph, pitchInt, n, k);
		cudaEventRecord(stepFinishedEvent);
		cudaEventSynchronize(stepFinishedEvent);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count() * 1000 << "(ms)\n";
	
	cudaStatus = cudaMemcpy2D(hostGraph, n * sizeof(int32_t), deviceGraph, pitch, n * sizeof(int32_t), n, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D failed!");
	}
	printf(check(hostGraph) ? "Result is correct\n" : "Result is not correct\n");
	writeResult(hostGraph);
	cudaFree(deviceGraph);
	return 0;
}

//Тайлинг второго уровня
__global__ void kernelI(int* __restrict__ deviceGraph, const int pitch, const int n, const int kBlock) {
	const int localI = threadIdx.y * R2;
	const int localJ = threadIdx.x * R2;
	const int base = kBlock * R1;
	//const int globalI = base + localI;
	//const int globalJ = base + localJ;
	__shared__ int localBlock[R1][R1];

	const int threadID = threadIdx.y * Q2 + threadIdx.x;
	const int offsetI = threadID / R1;
	const int offsetJ = threadID % R1;

	int i = offsetI;
//#pragma unroll
	for (int i1 = 0; i1 < S1; ++i1) {
		int j = offsetJ;
//#pragma unroll
		for (int i2 = 0; i2 < S2; ++i2) {
			localBlock[i][j] = deviceGraph[(base + i) * pitch + (base + j)];
			j += T2;
		}
		i += T1;
	}
	__syncthreads();
//#pragma unroll
	for (int k = 0; k < R1; ++k) {
//#pragma unroll
		for (int i = 0; i < R2; ++i) {
//#pragma unroll
			for (int j = 0; j < R2; ++j) {
				localBlock[localI + i][localJ + j] = min(localBlock[localI + i][localJ + j],
					localBlock[localI + i][k] + localBlock[k][localJ + j]);
			}
		}
		__syncthreads();
	}

	i = offsetI;
//#pragma unroll
	for (int i1 = 0; i1 < S1; ++i1) {
		int j = offsetJ;
//#pragma unroll
		for (int i2 = 0; i2 < S2; ++i2) {
			deviceGraph[(base + i) * pitch + (base + j)] = localBlock[i][j];
			j += T2;
		}
		i += T1;
	}
}

__global__ void kernelSD(int* __restrict__ deviceGraph, const int pitch, const int n, const int kBlock) {
	const int localI = threadIdx.y * R2;
	const int localJ = threadIdx.x * R2;
	const int baseLead = kBlock * R1;
	int baseI, baseJ;
	if (blockIdx.y == 0) {
		baseI = baseLead;
		baseJ = (blockIdx.x + (blockIdx.x >= kBlock)) * R1;
	}
	else {
		baseJ = baseLead;
		baseI = (blockIdx.x + (blockIdx.x >= kBlock)) * R1;
	}
	//const int globalI = baseI + localI;
	//const int globalJ = baseJ + localJ;
	__shared__ int localBlock[R1][R1];
	__shared__ int leadBlock[R1][R1];

	const int threadID = threadIdx.y * Q2 + threadIdx.x;
	const int offsetI = threadID / R1;
	const int offsetJ = threadID % R1;

	int i = offsetI;
//#pragma unroll
	for (int i1 = 0; i1 < S1; ++i1) {
		int j = offsetJ;
//#pragma unroll
		for (int i2 = 0; i2 < S2; ++i2) {
			localBlock[i][j] = deviceGraph[(baseI + i) * pitch + (baseJ + j)];
			j += T2;
		}
		i += T1;
	}

	i = offsetI;
//#pragma unroll
	for (int i1 = 0; i1 < S1; ++i1) {
		int j = offsetJ;
//#pragma unroll
		for (int i2 = 0; i2 < S2; ++i2) {
			leadBlock[i][j] = deviceGraph[(baseLead + i) * pitch + (baseLead + j)];
			j += T2;
		}
		i += T1;
	}
	__syncthreads();

	if (blockIdx.y == 0) {
//#pragma unroll
		for (int k = 0; k < R1; ++k) {
//#pragma unroll
			for (int i = 0; i < R2; ++i) {
//#pragma unroll
				for (int j = 0; j < R2; ++j) {
					localBlock[localI + i][localJ + j] = min(localBlock[localI + i][localJ + j],
						leadBlock[localI + i][k] + localBlock[k][localJ + j]);
				}
			}
			__syncthreads();
		}
	}
	else {
//#pragma unroll
		for (int k = 0; k < R1; ++k) {
//#pragma unroll
			for (int i = 0; i < R2; ++i) {
//#pragma unroll
				for (int j = 0; j < R2; ++j) {
					localBlock[localI + i][localJ + j] = min(localBlock[localI + i][localJ + j],
						localBlock[localI + i][k] + leadBlock[k][localJ + j]);
				}
			}
			__syncthreads();
		}
	}

	i = offsetI;
//#pragma unroll
	for (int i1 = 0; i1 < S1; ++i1) {
		int j = offsetJ;
//#pragma unroll
		for (int i2 = 0; i2 < S2; ++i2) {
			deviceGraph[(baseI + i) * pitch + (baseJ + j)] = localBlock[i][j];
			j += T2;
		}
		i += T1;
	}
}

__global__ void kernelDD(int* __restrict__ deviceGraph, const int pitch, const int n, const int kBlock) {
	const int localI = threadIdx.y * R2;
	const int localJ = threadIdx.x * R2;
	const int baseLead = kBlock * R1;
	const int baseI = (blockIdx.y + (blockIdx.y >= kBlock)) * R1;
	const int baseJ = (blockIdx.x + (blockIdx.x >= kBlock)) * R1;
	//const int globalI = baseI + localI;
	//const int globalJ = baseJ + localJ;
	__shared__ int leadRowBlock[R1][R1];
	__shared__ int leadColumnBlock[R1][R1];
	int c[R2][R2];

	const int threadID = threadIdx.y * Q2 + threadIdx.x;
	const int offsetI = threadID / R1;
	const int offsetJ = threadID % R1;

	int i = offsetI;
//#pragma unroll
	for (int i1 = 0; i1 < S1; ++i1) {
		int j = offsetJ;
//#pragma unroll
		for (int i2 = 0; i2 < S2; ++i2) {
			leadRowBlock[i][j] = deviceGraph[(baseI + i) * pitch + (baseJ + j)];
			j += T2;
		}
		i += T1;
	}
	__syncthreads();

//#pragma unroll
	for (int i = 0; i < R2; ++i) {
//#pragma unroll
		for (int j = 0; j < R2; ++j) {
			c[i][j] = leadRowBlock[localI + i][localJ + j];
		}
	}
	__syncthreads();
			
	i = offsetI;
//#pragma unroll
	for (int i1 = 0; i1 < S1; ++i1) {
		int j = offsetJ;
//#pragma unroll
		for (int i2 = 0; i2 < S2; ++i2) {
			leadRowBlock[i][j] = deviceGraph[(baseLead + i) * pitch + (baseJ + j)];
			j += T2;
		}
		i += T1;
	}

	i = offsetI;
//#pragma unroll
	for (int i1 = 0; i1 < S1; ++i1) {
		int j = offsetJ;
//#pragma unroll
		for (int i2 = 0; i2 < S2; ++i2) {
			leadColumnBlock[i][j] = deviceGraph[(baseI + i) * pitch + (baseLead + j)];
			j += T2;
		}
		i += T1;
	}
	__syncthreads();


//#pragma unroll
	for (int k = 0; k < R1; ++k) {
//#pragma unroll
		for (int i = 0; i < R2; ++i) {
//#pragma unroll
			for (int j = 0; j < R2; ++j) {
				c[i][j] = min(c[i][j], leadColumnBlock[localI + i][k] + leadRowBlock[k][localJ + j]);
			}
		}
	}
	__syncthreads();

/*
#pragma unroll
	for (int i2 = 0, i1 = localI; i2 < R2; ++i1, ++i2) {
#pragma unroll
		for (int j2 = 0, j1 = localJ; j2 < R2; ++j1, ++j2) {
			int ind = (globalI + i2) * pitch + (globalJ + j2);
			int c0 = deviceGraph[ind];
#pragma unroll
			for (int k = 0; k < R1; ++k) {
				c0 = min(c0, leadColumnBlock[i1][k] + leadRowBlock[k][j1]);
			}
			deviceGraph[ind] = c0;
		}
	}*/


#pragma unroll
	for (int i = 0; i < R2; ++i) {
#pragma unroll
		for (int j = 0; j < R2; ++j) {
			leadRowBlock[localI + i][localJ + j] = c[i][j];
		}
	}
	__syncthreads();

	i = offsetI;
#pragma unroll
	for (int i1 = 0; i1 < S1; ++i1) {
		int j = offsetJ; 
#pragma unroll
		for (int i2 = 0; i2 < S2; ++i2) {
			deviceGraph[(baseI + i) * pitch + (baseJ + j)] = leadRowBlock[i][j];
			j += T2;
		}
		i += T1;
	}
}

//Тайлинг первого уровня
/*__global__ void kernelI(int* __restrict__ deviceGraph, const int pitch, const int n, const int kBlock) {
	const int localI = threadIdx.y;
	const int localJ = threadIdx.x;
	const int base = kBlock * R1;
	const int globalI = base + localI;
	const int globalJ = base + localJ;
	__shared__ int localBlock[R1][R1];

	localBlock[localI][localJ] = deviceGraph[globalI * pitch + globalJ];
	__syncthreads();

#pragma unroll
	for (int k = 0; k < R1; ++k) {
		localBlock[localI][localJ] = min(localBlock[localI][localJ],
					localBlock[localI][k] + localBlock[k][localJ]);
		__syncthreads();
	}

	deviceGraph[globalI * pitch + globalJ] = localBlock[localI][localJ];
}

__global__ void kernelSD(int* __restrict__ deviceGraph, const int pitch, const int n, const int kBlock) {
	const int localI = threadIdx.y;
	const int localJ = threadIdx.x;
	const int baseLead = kBlock * R1;
	int baseI, baseJ;
	if (blockIdx.y == 0) {
		baseI = baseLead;
		baseJ = (blockIdx.x + (blockIdx.x >= kBlock)) * R1;
	}
	else {
		baseJ = baseLead;
		baseI = (blockIdx.x + (blockIdx.x >= kBlock)) * R1;
	}
	const int globalI = baseI + localI;
	const int globalJ = baseJ + localJ;
	__shared__ int localBlock[R1][R1];
	__shared__ int leadBlock[R1][R1];

	localBlock[localI][localJ] = deviceGraph[globalI * pitch + globalJ];
	leadBlock[localI][localJ] = deviceGraph[(baseLead + localI) * pitch + (baseLead + localJ)];
	__syncthreads();

	if (blockIdx.y == 0) {
#pragma unroll
		for (int k = 0; k < R1; ++k) {
			localBlock[localI][localJ] = min(localBlock[localI][localJ],
				leadBlock[localI][k] + localBlock[k][localJ]);
			__syncthreads();
		}
	}
	else {
#pragma unroll
		for (int k = 0; k < R1; ++k) {
			localBlock[localI][localJ] = min(localBlock[localI][localJ],
				localBlock[localI][k] + leadBlock[k][localJ]);
			__syncthreads();
		}
	}

	deviceGraph[globalI * pitch + globalJ] = localBlock[localI][localJ];
}

__global__ void kernelDD(int* __restrict__ deviceGraph, const int pitch, const int n, const int kBlock) {
	const int localI = threadIdx.y;
	const int localJ = threadIdx.x;
	const int baseLead = kBlock * R1;
	const int baseI = (blockIdx.y + (blockIdx.y >= kBlock)) * R1;
	const int baseJ = (blockIdx.x + (blockIdx.x >= kBlock)) * R1;
	const int globalI = baseI + localI;
	const int globalJ = baseJ + localJ;
	__shared__ int leadRowBlock[R1][R1];
	__shared__ int leadColumnBlock[R1][R1];

	int c = deviceGraph[globalI * pitch + globalJ];
	leadColumnBlock[localI][localJ] = deviceGraph[(baseI + localI) * pitch + (baseLead + localJ)];
	leadRowBlock[localI][localJ] = deviceGraph[(baseLead + localI) * pitch + (baseJ + localJ)];

	__syncthreads();

#pragma unroll
	for (int k = 0; k < R1; ++k) {
		c = min(c, leadColumnBlock[localI][k] + leadRowBlock[k][localJ]);
	}
	__syncthreads();

	deviceGraph[globalI * pitch + globalJ] = c;
}*/

//Тайлинг второго уровня с упрощённой щагрузкой из глобальной памяти
/*__global__ void kernelI(int* __restrict__ deviceGraph, const int pitch, const int n, const int kBlock) {
	const int localI = threadIdx.y * R2;
	const int localJ = threadIdx.x * R2;
	const int base = kBlock * R1;
	const int globalI = base + localI;
	const int globalJ = base + localJ;
	__shared__ int localBlock[R1][R1];

	for (int i = 0; i < R2; ++i) {
		for (int j = 0; j < R2; ++j) {
			localBlock[localI + i][localJ + j] = deviceGraph[(globalI + i) * pitch + (globalJ + j)];
		}
	}

	__syncthreads();
	//#pragma unroll
	for (int k = 0; k < R1; ++k) {
		//#pragma unroll
		for (int i = 0; i < R2; ++i) {
			//#pragma unroll
			for (int j = 0; j < R2; ++j) {
				localBlock[localI + i][localJ + j] = min(localBlock[localI + i][localJ + j],
					localBlock[localI + i][k] + localBlock[k][localJ + j]);
			}
		}
		__syncthreads();
	}

	//#pragma unroll
	for (int i = 0; i < R2; ++i) {
		for (int j = 0; j < R2; ++j) {
			deviceGraph[(globalI + i) * pitch + (globalJ + j)] = localBlock[localI + i][localJ + j];
		}
	}
}

__global__ void kernelSD(int* __restrict__ deviceGraph, const int pitch, const int n, const int kBlock) {
	const int localI = threadIdx.y * R2;
	const int localJ = threadIdx.x * R2;
	const int baseLead = kBlock * R1;
	int baseI, baseJ;
	if (blockIdx.y == 0) {
		baseI = baseLead;
		baseJ = (blockIdx.x + (blockIdx.x >= kBlock)) * R1;
	}
	else {
		baseJ = baseLead;
		baseI = (blockIdx.x + (blockIdx.x >= kBlock)) * R1;
	}
	const int globalI = baseI + localI;
	const int globalJ = baseJ + localJ;
	const int leadGlobalI = baseLead + localI;
	const int leadGlobalJ = baseLead + localJ;
	__shared__ int localBlock[R1][R1];
	__shared__ int leadBlock[R1][R1];

	for (int i = 0; i < R2; ++i) {
		for (int j = 0; j < R2; ++j) {
			localBlock[localI + i][localJ + j] = deviceGraph[(globalI + i) * pitch + (globalJ + j)];
		}
	}

	for (int i = 0; i < R2; ++i) {
		for (int j = 0; j < R2; ++j) {
			leadBlock[localI + i][localJ + j] = deviceGraph[(leadGlobalI + i) * pitch + (leadGlobalJ + j)];
		}
	}

	__syncthreads();

	if (blockIdx.y == 0) {
		//#pragma unroll
		for (int k = 0; k < R1; ++k) {
			//#pragma unroll
			for (int i = 0; i < R2; ++i) {
				//#pragma unroll
				for (int j = 0; j < R2; ++j) {
					localBlock[localI + i][localJ + j] = min(localBlock[localI + i][localJ + j],
						leadBlock[localI + i][k] + localBlock[k][localJ + j]);
				}
			}
			__syncthreads();
		}
	}
	else {
		//#pragma unroll
		for (int k = 0; k < R1; ++k) {
			//#pragma unroll
			for (int i = 0; i < R2; ++i) {
				//#pragma unroll
				for (int j = 0; j < R2; ++j) {
					localBlock[localI + i][localJ + j] = min(localBlock[localI + i][localJ + j],
						localBlock[localI + i][k] + leadBlock[k][localJ + j]);
				}
			}
			__syncthreads();
		}
	}

	for (int i = 0; i < R2; ++i) {
		for (int j = 0; j < R2; ++j) {
			deviceGraph[(globalI + i) * pitch + (globalJ + j)] = localBlock[localI + i][localJ + j];
		}
	}
}

__global__ void kernelDD(int* __restrict__ deviceGraph, const int pitch, const int n, const int kBlock) {
	const int localI = threadIdx.y * R2;
	const int localJ = threadIdx.x * R2;
	const int baseLead = kBlock * R1;
	const int baseI = (blockIdx.y + (blockIdx.y >= kBlock)) * R1;
	const int baseJ = (blockIdx.x + (blockIdx.x >= kBlock)) * R1;
	const int globalI = baseI + localI;
	const int globalJ = baseJ + localJ;
	const int leadGlobalI = baseLead + localI;
	const int leadGlobalJ = baseLead + localJ;
	__shared__ int leadRowBlock[R1][R1];
	__shared__ int leadColumnBlock[R1][R1];
	int c[R2][R2];

	//#pragma unroll
	for (int i = 0; i < R2; ++i) {
		//#pragma unroll
		for (int j = 0; j < R2; ++j) {
			c[i][j] = deviceGraph[(globalI + i) * pitch + (globalJ + j)];
		}
	}

	for (int i = 0; i < R2; ++i) {
		for (int j = 0; j < R2; ++j) {
			leadRowBlock[localI + i][localJ + j] = deviceGraph[(leadGlobalI + i) * pitch + (globalJ + j)];
		}
	}
	for (int i = 0; i < R2; ++i) {
		for (int j = 0; j < R2; ++j) {
			leadColumnBlock[localI + i][localJ + j] = deviceGraph[(globalI + i) * pitch + (leadGlobalJ + j)];
		}
	}

	__syncthreads();

	//#pragma unroll
	for (int k = 0; k < R1; ++k) {
		//#pragma unroll
		for (int i = 0; i < R2; ++i) {
			//#pragma unroll
			for (int j = 0; j < R2; ++j) {
				c[i][j] = min(c[i][j], leadColumnBlock[localI + i][k] + leadRowBlock[k][localJ + j]);
			}
		}
	}
	__syncthreads();

	//#pragma unroll
	for (int i = 0; i < R2; ++i) {
		//#pragma unroll
		for (int j = 0; j < R2; ++j) {
			deviceGraph[(globalI + i) * pitch + (globalJ + j)] = c[i][j];
		}
	}
}*/	
