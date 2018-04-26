//(A*X + B)mod N
#include <stdio.h>
#include "cycleTimer.h"

#define A 5
#define B 7
#define N 10000
#define ARRAY_LENGTH 7168//81920 //1024*80
#define ITER 200000
#define THREADS_PER_BLOCK 128


__global__ void kernel(int* num)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= ARRAY_LENGTH) {
		return;
	}

	int X = threadId;
	
	for (int i = 0; i < ITER; i++) {
		X = (A*X + B) % N;
	}
	num[threadId] = X;
}

void sequential(int *num) {
	for (int n = 0; n < ARRAY_LENGTH; n++) {
		int X = n;
		
		for (int i = 0; i < ITER; i++) {
			X = (A*X + B) % N;
		}
		num[n] = X;
	}
}

bool checkCorrectness(int *h_numbers, int *seq_numbers) {
	for (int i = 0; i < ARRAY_LENGTH; i++) {
		if (h_numbers[i] != seq_numbers[i]) {
			return false;
		}
	}
	return true;
}

int main()
{
	int *d_numbers, *h_numbers, *seq_numbers;
	double startTime, endTime, overallDuration;

	//malloc cuda and sequential
	cudaMalloc(&d_numbers, sizeof(int) * ARRAY_LENGTH);
	h_numbers = (int*)malloc(sizeof(int)*ARRAY_LENGTH);
	seq_numbers = (int*)malloc(sizeof(int)*ARRAY_LENGTH);
	
	int blocks = (ARRAY_LENGTH + THREADS_PER_BLOCK- 1) / THREADS_PER_BLOCK;
	unsigned int count = 0;
		
	while(1) {
		startTime = CycleTimer::currentSeconds();
		kernel<<<blocks, THREADS_PER_BLOCK>>>(d_numbers);
		cudaThreadSynchronize();
		endTime = CycleTimer::currentSeconds();
		
		overallDuration = endTime - startTime;
		if (count++ % 100 == 0) {
			if (1000.f * overallDuration > 4.5) {
				printf("Deadline missed ");
			}
			printf("Overall parallel: %.3f ms\n", 1000.f * overallDuration);
		}
		
	}

	cudaMemcpy(h_numbers, d_numbers, sizeof(double)*ARRAY_LENGTH, cudaMemcpyDeviceToHost);

	//startTime = CycleTimer::currentSeconds();
	//sequential(seq_numbers);
	//endTime = CycleTimer::currentSeconds();
	//overallDuration = endTime - startTime;
	//printf("Overall serial: %.3f ms\n", 1000.f * overallDuration);

	//if(checkCorrectness(h_numbers, seq_numbers)) {
	//	printf("correctness passed\n");
	//} else {
	//	printf("correctness failed\n");
	//}


	//cleanup
	cudaFree(d_numbers);
	free(h_numbers);
	free(seq_numbers);
	

	return 0;
}

