#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "CPU_version.h"


using namespace std;

#define BLOCKSIZE 256;

float LOG2(float N)
{
	return log(N)/log(2.0f);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////INCLUSIVE & EXCLUSIVE PREFIX SUM////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void inclusivePrefixSum(float *d_input, float *d_output, int N, int d_offset)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= N)
		return;

	d_output[index] = d_input[index];
	
	__syncthreads();
	
	if(index >= d_offset)
	{
		d_input[index] = d_output[index - d_offset] + d_output[index];
	}
}

__global__ void toExclusive(float *inclusive_array, float *exclusive_array, int N) 
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (index == 0)	
		exclusive_array[index] = 0; 
	else
		exclusive_array[index] = inclusive_array[index - 1]; 

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////NAIVE PARALLEL SCAN/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
void naiveParallelScan(float *d_input, float *d_output, int N)
{
	
	int blockSize = BLOCKSIZE;
	int gridSize = ceil((float)N/(float)blockSize); 

	dim3 numberOfBlocks(gridSize);
	dim3 threadsPerBlock(blockSize);
	
	for(int offset = 1; offset <= LOG2(N) + 1; offset++)
	{
		int d_offset = pow(2.0f, offset - 1);
		inclusivePrefixSum<<<numberOfBlocks,threadsPerBlock>>>(d_input, d_output, N, d_offset);
		toExclusive<<<numberOfBlocks,threadsPerBlock>>>(d_input, d_output, N);
	}

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////SINGLE BLOCK SCAN/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void singleBlockPrefixSum(float * d_input,float * d_output, int N, int d_offset)
{
	extern __shared__ float shared_array[];	
	int index = threadIdx.x;

	if (index >= N)
		return;
	
	shared_array[index] = d_input[index];
	
	__syncthreads();


	if(index >= d_offset)
	{
		d_input[index] = shared_array[index - d_offset] + shared_array[index];
	}

	__syncthreads();
	
	d_output[index] = (index>0) ? d_input[index-1] : 0.0f;
}

void singleBlockScan(float *d_input, float *d_output, int N)
{
	for(int offset = 1; offset < LOG2(N) + 1; offset++)
	{
		int d_offset = pow(2.0f, offset - 1);
		singleBlockPrefixSum<<<1, N + 1, N * sizeof(float)>>>(d_input, d_output, N, d_offset);
	}

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////MULTI BLOCK PREFIX SUM/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void addIncs(float *d_input, float *inter, float *d_output, int N)
{
	int globalIndex = (blockDim.x * blockIdx.x) + threadIdx.x;
	if(globalIndex >= N + 1)
		return;

	if(blockIdx.x >= 1) 
		d_input[globalIndex] = d_input[globalIndex] + inter[blockIdx.x - 1];
	
	if(globalIndex < N + 1)	
		d_output[globalIndex] = (globalIndex < 1) ? 0.0f : d_input[globalIndex - 1];

}
__global__ void generalPrefixSum(float * d_input, float *inter, int N, int d_offset)
{

	extern __shared__ float shared_array[];	
	int localIndex = threadIdx.x;
	int globalIndex = (blockDim.x * blockIdx.x) + threadIdx.x;

	if (globalIndex >= N)
		return;
	
	shared_array[localIndex] = d_input[globalIndex];
	
	__syncthreads();


	if(localIndex >= d_offset)
	{
		d_input[globalIndex] = shared_array[localIndex - d_offset] + shared_array[localIndex];
	}

	__syncthreads();

	if(localIndex ==  blockDim.x - 1)
		 inter[blockIdx.x] = (globalIndex > N) ? d_input[N - 1]: d_input[globalIndex];

}

void parallelScan(float *input, float *output, int N)
{
	int threadsPerBlock = BLOCKSIZE;
	int numberOfBlocks = ceil((float)N/(float)threadsPerBlock);

	float *d_input, *inter_sum, *d_temp;
	
	cudaMalloc((void**) &d_input, N * sizeof(float));
	cudaMalloc((void**) &inter_sum, (numberOfBlocks) * sizeof(float));
	cudaMalloc((void**) &d_temp, (N + 1) * sizeof(float));

	cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyDeviceToDevice);

	for(int offset = 1; offset < LOG2(N) + 1; offset++)
	{
		int d_offset = pow(2.0f, offset - 1);
		generalPrefixSum<<<numberOfBlocks, threadsPerBlock, N * sizeof(float)>>>(d_input, inter_sum, N, d_offset);
	}

	for(int offset = 1; offset < LOG2(numberOfBlocks) + 1; offset++)
	{
		int d_offset = (int)pow(2.0f, offset - 1);
		inclusivePrefixSum<<<ceil((float)numberOfBlocks/(float)threadsPerBlock), threadsPerBlock, numberOfBlocks * sizeof(float)>>>(inter_sum, d_temp, numberOfBlocks, d_offset);
	}
	
	addIncs<<<numberOfBlocks + 1, threadsPerBlock>>>(d_input, inter_sum, output, N+1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////STREAM COMPACTION/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void createBooleanArray(float *d_input, float *d_output, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index >= N)
		return;
		
	if (d_input[index] == 0.0f)
		d_output[index] = 0.0f; 
	else 
		d_output[index] = 1.0f;
}

__global__ void compactedArray(float *d_input, float *d_output, float *boolean_array, float * index_array, int N)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index >= N)
		return;
	
	if(boolean_array[index] != 0.0f) 
		d_output[(int)index_array[index]] = d_input[index];
	
}

void streamCompaction(float *d_input, float *d_output, int N)
{
	int threadsPerBlock = BLOCKSIZE;
	int numberOfBlocks = ceil(( float )N / ( float )threadsPerBlock);

	float *boolean_array, *index_array;
	cudaMalloc((void**) &boolean_array, N * sizeof(float));
	cudaMalloc((void**) &index_array, (N + 1) * sizeof(float));

	createBooleanArray<<<numberOfBlocks, threadsPerBlock>>>(d_input, boolean_array, N);
	parallelScan(boolean_array, index_array, N);
	compactedArray<<<numberOfBlocks, threadsPerBlock>>>(d_input, d_output, boolean_array, index_array, N);

}

struct notZero
{
    __host__ __device__ 
	bool operator()(const float x)
    {
      return x != 0;
    }
 };

void streamCompactionThrust(float *input, float *output, int N)
{
	double dtime = omp_get_wtime();

	thrust::copy_if(input, input + N, output, notZero());

	dtime = omp_get_wtime() - dtime;
	std::cout << "Stream Compaction Thrust Runtime: " << dtime * 1000 << " ms." << std::endl;
}

int main()
{

	int numObject = 100;
	
	//Timers
	cudaEvent_t start,stop;
	float time = 0.0f; 

	//Input and output arrays
	float *h_input = new float[numObject]; 
	float *h_output = new float[numObject]; 
		
	//Populate input array
	for(int i = 0; i < numObject; ++i)
	{
		
		h_input[i] = 1;
		/*if (i % 3 == 0 )
			h_input[i] = 0;
		else
			h_input[i] = i;
			*/
		//std::cout<<h_input[i] << std::endl;
	}

	
	float *d_input, *d_output;
	
	//Allocate cuda memory
	cudaMalloc((void**) &d_input, numObject * sizeof(float));
	cudaMalloc((void**) &d_output, (numObject + 1) * sizeof(float));
	
	//Copy memory from host to device
	cudaMemcpy(d_input, h_input, numObject * sizeof(float), cudaMemcpyHostToDevice);
	
	//Start timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Device functions
	//naiveParallelScan(d_input, d_output, numObject);
	//parallelScan(d_input, d_output, numObject);
	//streamCompaction(d_input, d_output, numObject);
	
	//Stop timers
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cout<< "Runtime: " << time << " ms." << endl;

	//Copy memory from device to host
	cudaMemcpy(h_output, d_output, numObject * sizeof(float), cudaMemcpyDeviceToHost);
	
	
	//Host functions
	//serialPrefixSum(h_input, h_output, numObject);
	//streamCompactionThrust(h_input, h_output, numObject);
  
	for(int i = 0; i < numObject; ++i)
	{
		//cout << h_output[i]<<" ";
	}

	cudaFree(d_input);
	cudaFree(d_output);

	system("pause");
    return 0;
}