/* Generate some simple pattern to test OpenGL-Cuda interop.
*/

//cuda
#include <cuda_runtime.h>
#include "3rdParty/cuda-helper/helper_cuda.h"

//misc
#include "util/global.h"
#include <cmath>

surface<void, 2> surfD;

__global__ void doTransform(dim3 N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= N.x || j >= N.y)
		return;

	float height = (sin((float)i *0.02)*10.0) * (sin((float)j*0.02)*10.0);
	surf2Dwrite(make_float4(height,.0 ,.0,.0)
	, surfD, i*sizeof(float4), j);
}

extern "C" void runGenerate(cudaArray *img, size_2D size)
{

	/* Get maximum threads per block */
	int dev;
	checkCudaErrors(cudaGetDevice(&dev));
	cudaDeviceProp prop; 
	checkCudaErrors(cudaGetDeviceProperties(&prop,dev)); 
	int threadsPerBlockDim = floor(sqrt((float)prop.maxThreadsPerBlock)); 
	dim3 threadsPerBlock;
	threadsPerBlock.x = threadsPerBlock.y = threadsPerBlockDim;
	/*Round up number of Blocks, then check for
	* valid index inside of kernels */
	dim3 numBlocks(ceil((float)size.x/threadsPerBlock.x), ceil((float)size.x/threadsPerBlock.y));
	dim3 n;
	n.x = size.x;
	n.y = size.z;
	cudaChannelFormatDesc desc;
	checkCudaErrors(cudaGetChannelDesc(&desc,img));

	checkCudaErrors(cudaBindSurfaceToArray(surfD,img));

	/*//check array
	float4* hostBuf = new float4[100*100];
	size_t pitch = 100 * sizeof(float4);
	checkCudaErrors(cudaMemcpy2DFromArray(hostBuf,pitch, img, 0, 0, pitch,
		100, cudaMemcpyDeviceToHost));*/
	//
	doTransform <<< numBlocks,threadsPerBlock >>> (n);
	checkCudaErrors(cudaGetLastError());

	//vheckCudaErrors(cudaMemcpy2DFromArray(hostBuf,pitch, img, 0, 0, pitch,
	//	100, cudaMemcpyDeviceToHost));
}


