cuda_kernel: cuda_kernel.cu cycleTimer.h
	nvcc -o cuda_kernel cuda_kernel.cu
