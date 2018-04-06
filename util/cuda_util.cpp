#include "util\cuda_util.h"
dim3 threadsPerBlock;
dim3 getThreadsPerBlock2D()
{
	static bool init = true;

	if( init )
	{
		/* Get maximum threads per block */
		int dev;
		checkCudaErrors(cudaGetDevice(&dev));
		cudaDeviceProp prop; 
		checkCudaErrors(cudaGetDeviceProperties(&prop,dev)); 
		int threadsPerBlockDim = (int)floor(sqrt((float)prop.maxThreadsPerBlock)); 

		//threadsPerBlock.x = threadsPerBlock.y //threadsPerBlockDim;
		threadsPerBlock.x = BLOCK_SIZE_X;
		threadsPerBlock.y = BLOCK_SIZE_Y;
		threadsPerBlock.z = 1;
		init = false;
	}
	return threadsPerBlock;
}