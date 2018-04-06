#include "util\cuda_util.h"
#include "util\general_functions.cuh"

#include <cuda_runtime.h>

__global__ void multiply2DScalar(float factor ,float* data , size_t pitch,  dim3 N)
{
	DEFINE_DEFAULT_KERNEL_WORKID_2D(i,j);
	RETURN_IF_OUTSIDE_2D(i,j,N);

	float* elPtr = (float*)((char*)data + i * pitch) + j;
	*elPtr *= factor;
}

//add value of inc to field ( per cell )
__global__ void increment2D_2D( float *field , size_t fieldPitch,  
								float *inc, size_t incPitch, dim3 N)
{
	DEFINE_DEFAULT_KERNEL_WORKID_2D(i,j);
	RETURN_IF_OUTSIDE_2D(i,j,N);

	float* fieldPtr = (float*)((char*)field + i * fieldPitch) + j;
	float* incPtr = (float*)((char*)inc + i * incPitch) + j;

	*fieldPtr += *incPtr;
}

__global__ void memsetFloat2D( float* field, size_t pitch, float value,  dim3 N)
{
	DEFINE_DEFAULT_KERNEL_WORKID_2D(i,j);
	RETURN_IF_OUTSIDE_2D(i,j,N);

	float* ptr = (float*)((char*)field + i * pitch) + j;
	*ptr = value;
}


extern void cw_memsetFloat2D(floatMem field, size_2D size,  float value)
{
	memsetFloat2D <<< getNumBlocks2D(size) , getThreadsPerBlock2D() >>>
		(field.devPtr,field.pitch, value, dim3FromSize_2D(size));
}