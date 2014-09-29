#include <Windows.h>
#include <iostream>
#include <omp.h>



void serialPrefixSum(float *input, float* output, int N);
void serialCreateBooleanArray(float *input, float *output, int i);
void serialCompactedArray(float *input, float *output, float *boolean_array, float *index_array, int i);
void serialStreamCompaction(float *input, float* output, int N);