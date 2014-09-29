#include "CPU_version.h"



void serialPrefixSum(float *input, float* output, int N)
{
	double dtime = omp_get_wtime();
  		
	int x = 0;

	for (int i = 0; i < N; ++i)
	{
		output[i] = x;
		x = x + input[i];

	}
	dtime = omp_get_wtime() - dtime;
	std::cout << "CPU Prefix Sum Runtime: " << dtime * 1000 << " ms." << std::endl;
}

void serialCreateBooleanArray(float *input, float *output, int i)
{
	if (input[i] == 0.0f)
		output[i] = 0.0f; 
	else 
		output[i] = 1.0f;
}

void serialCompactedArray(float *input, float *output, float *boolean_array, float *index_array, int i)
{
	if(boolean_array[i] != 0.0f) 
		output[(int)index_array[i]] = input[i];

}
void serialStreamCompaction(float *input, float* output, int N)
{
	
	float *boolean_array = new float[N];
	float *index_array = new float[N + 1];

	for (int i = 0; i < N; ++i)
	{
		serialCreateBooleanArray(input, boolean_array, i);
	}

	for(int i = 0; i <= N; ++i)
	{
		serialCompactedArray(input, output, boolean_array, index_array, i);
	}

}

