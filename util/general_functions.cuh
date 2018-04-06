#ifndef GENERAL_FUNCTIONS_CUH
#define GENERAL_FUNCTIONS_CUH

#include <cuda_runtime.h>

__global__ void multiply2DScalar(float factor ,float* data , size_t pitch,  dim3 N);


//add value of inc to field ( per cell )
__global__ void increment2D_2D( float *field , size_t fieldPitch,  
								float *inc, size_t incPitch, dim3 N);

__global__ void memsetFloat2D( float* field, size_t pitch, float value , dim3 N);


#endif //GENERAL_FUNCTIONS_CUH