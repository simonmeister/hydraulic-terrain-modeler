#ifndef CUDA_INFO_H
#define CUDA_INFO_H

#include <cuda_runtime.h>
#include <cmath>
#include "global.h"


/////////////////////
#ifdef IS_DEBUG
#include "3rdParty\cuda-helper\helper_cuda.h"
#else
#undef checkCudaErrors
#define checkCudaErrors( X ) X
#endif

////////////////////

/* Information about execution domain*/

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
dim3 getThreadsPerBlock2D();
extern dim3 threadsPerBlock;
inline dim3 getNumBlocks2D( size_2D size )
{
	dim3 nb = getThreadsPerBlock2D();
	return dim3((unsigned int) ceil( (float) size.rows/nb.x )
		      , (unsigned int) ceil( (float) size.cols/nb.y ));
}
/*
inline dim3 getNumBlocks2D( size_2D size , size_t blockSize )
{
	dim3 nb(blockSize,blockSize,1);
	return dim3((unsigned int) ceil( (float) size.rows/nb.x )
		      , (unsigned int) ceil( (float) size.cols/nb.y ));
}
*/
inline dim3 dim3FromSize_2D( size_2D size )
{ 
	dim3 res;
	res.x = size.rows;
	res.y = size.cols;
	res.z = 0;
	return res;
}

inline dim3 dim3Uni( size_t unif )
{ 
	dim3 res;
	res.x = unif;
	res.y = unif;
	res.z = unif;
	return res;
}



/* Typedefs and structs */
template <typename T>
struct memCuda2D
{
	T* devPtr;
	size_t pitch;
};

typedef memCuda2D<float> floatMem;
typedef memCuda2D<bool> byteMem;
typedef memCuda2D<int> intMem;



#define SAFE_CUDA_FREE( X ) \
	if( X ){ checkCudaErrors(cudaFree(X)) ; X = nullptr; }

/*Macros to be used inside of CUDA device code */
//the repetitive code to get work item IDs can be replaced by this macro
#define DEFINE_DEFAULT_KERNEL_WORKID_2D( DIMX_NAME, DIMY_NAME) \
	int DIMX_NAME = blockIdx.x * blockDim.x + threadIdx.x;	   \
	int DIMY_NAME = blockIdx.y * blockDim.y + threadIdx.y

#define RETURN_IF_OUTSIDE_2D(DIMX_ID, DIMY_ID, N) \
	if(DIMX_ID >= N.x || DIMY_ID >= N.y)		  \
		return

#define KERNEL_INIT_2D() \
	DEFINE_DEFAULT_KERNEL_WORKID(i, j) \
	RETURN_IF_OUTSIDE(i,j, N)

#endif //CUDA_INFO_H