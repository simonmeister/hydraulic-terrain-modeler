#include <cuda_runtime.h>

#include "3rdParty\cuda-helper\helper_cuda.h"
#include "util\global.h"
#include "util\cuda_util.h"
#include "sourcing_types.h"

//------------------------------------------------------------------------
// Radial 
//------------------------------------------------------------------------


__global__ void addSourcingToSubfieldRadial( float* origField ,size_t origPitch, size_2D origSize , 
									 float scalar, size_2D subOffset, 
									 float radius, size_2D srcPos,
									 float hardness, bool falloff, dim3 N)                                
{
	DEFINE_DEFAULT_KERNEL_WORKID_2D(i,j);
	RETURN_IF_OUTSIDE_2D(i,j,N);

	size_t origFieldIdxX = i+ subOffset.x;
	size_t origFieldIdxZ = j+ subOffset.z;

	//measure length between midIdx and current element as int-indices
	int diffx = (srcPos.x - origFieldIdxX);
	int diffz = (srcPos.z - origFieldIdxZ);

	//transform back to 0...1 and compute vector length 
	//preserve aspect ratio!
	float dist = 0.0f;
	if(origSize.x < origSize.z )
	{
		dist = sqrt( pow((diffx / (float)(origSize.x-1)),2)  +  
			pow((diffz /(float)(origSize.z-1)) *(origSize.z / origSize.x),2) );
	}
	else
	{
		dist = sqrt( pow((diffx / (float)(origSize.x-1))*(origSize.x / origSize.z),2)  +  
			pow(diffz /(float)(origSize.z-1) ,2) );
	}

	if( dist > radius)
		return;


	float *cell = (float*)((char*)origField + origFieldIdxX * origPitch) + origFieldIdxZ;

	if(falloff)
	{
		*cell = ( *cell + scalar > 0.0f ? *cell + 
			scalar * ( hardness * (1.0 - ( dist/radius)) )
			: 0.0f);
	}
	else
	{
		*cell = ( *cell + scalar > 0.0f ? *cell + scalar : 0.0f);
	}

}

extern void cw_radialSourcing(floatMem& heights, size_2D gridSize,
	                        float perCellAmount, 
							size_2D fromIdx, size_2D toIdx, size_2D posIdx,
							float radius, float hardness, bool useFalloff)
{
	size_2D size;
	size.x = (toIdx.x + 1) - fromIdx.x;
	size.z = (toIdx.z + 1) - fromIdx.z;

	
	addSourcingToSubfieldRadial <<< getNumBlocks2D(size), getThreadsPerBlock2D() >>>
		(heights.devPtr, heights.pitch, gridSize,
		perCellAmount, fromIdx,radius ,posIdx,hardness, useFalloff,
		dim3FromSize_2D(size));
	checkCudaErrors(cudaGetLastError());
}

//------------------------------------------------------------------------
// Rectangular
//------------------------------------------------------------------------

__global__ void addSourcingToSubfield( float* origField ,size_t origPitch,
									 float scalar, size_2D subOffset, dim3 N)                                
{
	DEFINE_DEFAULT_KERNEL_WORKID_2D(i,j);
	RETURN_IF_OUTSIDE_2D(i,j,N);

	float *cell = (float*)((char*)origField + 
		(i+ subOffset.x) * origPitch) + j + subOffset.z;


	
	//avoid negative heights if its a sink
	*cell = ( *cell + scalar > 0.0f ? *cell + scalar : 0.0f);

}
extern void cw_rectSourcing(floatMem& heights, size_2D gridSize,
	                        float perCellAmount, 
							size_2D fromIdx, size_2D toIdx)
{
	size_2D size;
	size.x = (toIdx.x + 1) - fromIdx.x;
	size.z = (toIdx.z + 1) - fromIdx.z;

	addSourcingToSubfield <<< getNumBlocks2D(size), getThreadsPerBlock2D() >>>
		(heights.devPtr, heights.pitch,
		perCellAmount, fromIdx , dim3FromSize_2D(size));
	checkCudaErrors(cudaGetLastError());
}