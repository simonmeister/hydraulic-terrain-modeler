#include <cuda_runtime.h>
#include <vector>

#include "3rdParty\cuda-helper\helper_cuda.h"
#include "util\global.h"
#include "util\cuda_util.h"

#include "WaterPipeSimulator_types.h"




__constant__ float sc[2];

extern void cw_setConstant(WaterPipeSimulatorConst c, float value)
{
	checkCudaErrors(cudaMemcpyToSymbol(sc,&value,sizeof(float), sizeof(float) * c));
}

//------------------------------------------------------------------------
// Initialize velocities as zero
//------------------------------------------------------------------------
__global__ void threadWetStates( float* waterHeights, size_t waterPitch,
						       int* activeBlocks, size_t abPitch,
							   dim3 N)
{
	DEFINE_DEFAULT_KERNEL_WORKID_2D(j,i);
	RETURN_IF_OUTSIDE_2D(j,i,N);

	float* waterPtr = (float*)((char*)waterHeights + i * waterPitch) + j;

	__shared__ int* abPtr;
	//init abPtr once per block
	if(threadIdx.x == 0 && threadIdx.y == 0)
		abPtr = (int*)((char*)activeBlocks + blockIdx.y * abPitch) + blockIdx.x;
	__syncthreads();

	if(*waterPtr > 0.0)
		atomicAdd(abPtr, 1);
}

__global__ void blocksNeighbours( bool* activeBlocksResult, size_t abResPitch,
	                              int* activeBlocksTemp, size_t abTempPitch, dim3 N )
{
	DEFINE_DEFAULT_KERNEL_WORKID_2D(j,i);
	RETURN_IF_OUTSIDE_2D(j,i,N);

	bool* meBoolPtr = (bool*)((char*)activeBlocksResult + i * abResPitch) + j;
	int* meIntPtr = (int*)((char*)activeBlocksTemp + i * abTempPitch) + j;
	//Order bottom, top, right, left
	bool validDirect[4];

	//Order: bottom left, bottom right, top left, top right
	bool validCorners[4];
	//////////
	validDirect[0] = validDirect[1] = validDirect[2] = validDirect[3] = true;
	validCorners[0] = validCorners[1] = validCorners[2] = validCorners[3] = true;
	
	if( i == (N.y-1) )
		validDirect[0] = false;
	else if( i == 0)
		validDirect[1] = false;
	
	if( j == (N.x -1) )
		validDirect[2] = false;
	else if( j == 0)
		validDirect[3] = false;
	////////////////
	if(i== (N.y-1) || j == 0)
		validCorners[1] = false;
	if(i== (N.y-1) || j == (N.x -1))
		validCorners[0] = false;
	if(i== 0 || j == 0)
		validCorners[2] = false;
	if(i== 0 || j ==  (N.x -1))
		validCorners[3] = false;

	bool wetThis = *meIntPtr > 0;

	if(!wetThis)
	{	//use else if as it is enough to set it once
		if( validDirect[0] &&      *(	(int*)((char*)activeBlocksTemp + (i+1) * abTempPitch) + j   ) > 0) 
			wetThis = true;
		else if(validDirect[1] &&  *(	(int*)((char*)activeBlocksTemp + (i-1) * abTempPitch) + j   ) > 0) 
			wetThis = true;
		else if(validDirect[2] &&  *(	(int*)((char*)activeBlocksTemp + i * abTempPitch) + (j+1)   ) > 0)
			wetThis = true;
		else if(validDirect[3] &&  *(	(int*)((char*)activeBlocksTemp + i * abTempPitch) + (j-1)   ) > 0)
			wetThis = true;

		else if( validCorners[0] &&      *(	(int*)((char*)activeBlocksTemp + (i+1) * abTempPitch) + j-1   ) > 0) 
			wetThis = true;
		else if(validCorners[1] &&  *(	(int*)((char*)activeBlocksTemp + (i+1) * abTempPitch) + j+1   ) > 0) 
			wetThis = true;
		else if(validCorners[2] &&  *(	(int*)((char*)activeBlocksTemp + (i-1) * abTempPitch) + (j-1)   ) > 0)
			wetThis = true;
		else if(validCorners[3] &&  *(	(int*)((char*)activeBlocksTemp + (i-1) * abTempPitch) + (j+1)   ) > 0)
			wetThis = true;
	}
	*meBoolPtr = wetThis;
}

__global__ void blocksInit(  int* activeBlocks, size_t abPitch, dim3 N )
{
	DEFINE_DEFAULT_KERNEL_WORKID_2D(j,i);
	RETURN_IF_OUTSIDE_2D(j,i,N);

	int* ptr = (int*)((char*)activeBlocks + i * abPitch) + j;
	*ptr = 0;

}
extern void cw_blockActivity( floatMem& waterHeights, byteMem& activeBlocksResult, intMem& activeBlocksTemp,
							 size_2D numAb, size_2D size)
{
	size_2D numAbInv(numAb.z,numAb.x);
	//initialize each blocks active state with 0
	blocksInit <<< getNumBlocks2D(numAbInv), getThreadsPerBlock2D() >>>
		(activeBlocksTemp.devPtr, activeBlocksTemp.pitch, dim3FromSize_2D(numAbInv));

	checkCudaErrors(cudaGetLastError());

	size_2D sizeInv(size.z,size.x);
	//execute per thread, adding to current blocks active state if wet
	threadWetStates <<< getNumBlocks2D(sizeInv), getThreadsPerBlock2D() >>>
		(waterHeights.devPtr, waterHeights.pitch
		, activeBlocksTemp.devPtr, activeBlocksTemp.pitch, dim3FromSize_2D(sizeInv));

	checkCudaErrors(cudaGetLastError());
	
	//evaluate self and neighbours wet states to get the final map
	blocksNeighbours <<< getNumBlocks2D(numAbInv), getThreadsPerBlock2D() >>>
		(activeBlocksResult.devPtr, activeBlocksResult.pitch,
		activeBlocksTemp.devPtr, activeBlocksTemp.pitch,
		dim3FromSize_2D(numAbInv));

	checkCudaErrors(cudaGetLastError());
}

//------------------------------------------------------------------------
// Set all blocks to "active"
//------------------------------------------------------------------------

__global__ void blocksSetAllActive(bool* field, size_t pitch, dim3 N)
{
	DEFINE_DEFAULT_KERNEL_WORKID_2D(i,j);
	RETURN_IF_OUTSIDE_2D(i,j,N);

	bool* ptr = (bool*)((char*)field + i * pitch) + j;
	*ptr = true;
}
extern void cw_blocksSetAllActive( byteMem& activeBlocks, size_2D size)
{
	blocksSetAllActive <<< getNumBlocks2D(size), getThreadsPerBlock2D()  >>>
		(activeBlocks.devPtr, activeBlocks.pitch, dim3FromSize_2D(size));

}

//------------------------------------------------------------------------
// Initialize velocities as zero
//------------------------------------------------------------------------

__global__ void memsetCell4( cell4* field, size_t pitch, float value,  dim3 N)
{
	DEFINE_DEFAULT_KERNEL_WORKID_2D(i,j);
	RETURN_IF_OUTSIDE_2D(i,j,N);

	cell4* ptr = (cell4*)((char*)field + i * pitch) + j;
	ptr->B = 0.0;
	ptr->T = 0.0;
	ptr->R = 0.0;
	ptr->L = 0.0;
}

extern void cw_memsetCell4(float value, const cell4Mem& repr, size_2D size)
{
	memsetCell4 <<< getNumBlocks2D(size), getThreadsPerBlock2D() >>>
		(repr.devPtr,repr.pitch,value,dim3FromSize_2D(size));
	checkCudaErrors(cudaGetLastError());
}

//------------------------------------------------------------------------
// constant storage for boundary settings
//------------------------------------------------------------------------

__constant__ boundCond boundaries[4];

extern void writePipeBounds( boundCond* conds)
{
	checkCudaErrors(cudaMemcpyToSymbol(boundaries,conds,sizeof(boundaries)));
}

//------------------------------------------------------------------------
// calculate velocities for next step
//------------------------------------------------------------------------

#define GRAVITY_CONSTANT 9.81

__global__ void positiveVelocitiesBTRL( cell4* velField, size_t velPitch, 
										float* waterField, size_t waterPitch,
										layer* layers, unsigned int layerCount,
										bool* activeBlocks, size_t abPitch , 
										float* thStorage, size_t thsPitch,
										dim3 N)
{
	bool* abPtr = (bool*)((char*)activeBlocks + blockIdx.y * abPitch) + blockIdx.x;
	if(!*abPtr)
		return;

	DEFINE_DEFAULT_KERNEL_WORKID_2D(j,i);
	RETURN_IF_OUTSIDE_2D(j,i,N);

	float* waterPtr = (float*)((char*)waterField + i * waterPitch) + j;
	//normalize waterheight (for block deactivations)
	*waterPtr = (*waterPtr <= sc[CONSTANT_WATERDRY_THRESHOLD]) ? 0.0 : *waterPtr;

	//-----------------------
	// useful variables 
	//-----------------------

	size_t localRow = threadIdx.y;
	size_t localCol = threadIdx.x;

	cell4* velPtr = (cell4*)((char*)velField + i * velPitch) + j;

	float neighbourHeight;
	float* neighbourPtr;

	/*
	The first branching does not impact performance, as it almost never evaluates to true.
	For the second level of branching, 2/3 of all threads branch to use shared memory. 
	This use of shared memory doubles execution speed of this kernel */


	//-----------------------
	// Sum up this cells water height and layer heights
	// Use shared memory as nearby threads also need this information
	//-----------------------

	__shared__ float totalHeight[BLOCK_SIZE_Y][BLOCK_SIZE_X];
	float thisHeight = *waterPtr;
	float* tmp;
	for(int f = 0; f < layerCount; ++f)
	{
		tmp = (float*)((char*)layers[f].ptr + i * layers[f].pitch) + j;
		thisHeight += *tmp;
	}

	totalHeight[localRow][localCol] = thisHeight;
	//write to buffer
	*((float*)((char*)thStorage + i * thsPitch) + j) = thisHeight;
	//now sync as next steps will need neighbouring height values
	__syncthreads();

	float precompMultiplier = GRAVITY_CONSTANT * sc[CONSTANT_TIMESTEP];

	//-----------------------
	// Top pipe
	//-----------------------
	if( i == 0 )
	{
		if(boundaries[1].reflect)
			neighbourHeight = thisHeight;
		else
			neighbourHeight = boundaries[1].level;
	}
	else
	{
		if( threadIdx.y == 0 )
		{
			//water
			neighbourPtr = (float*)((char*)waterField + (i-1) * waterPitch) + j;
			neighbourHeight = *neighbourPtr;
			//sum up layers
			for(int f = 0; f < layerCount; ++f)
			{
				tmp = (float*)((char*)layers[f].ptr + (i-1) * layers[f].pitch) + j;
				neighbourHeight += *tmp;
			}
		}
		else
		{
			neighbourHeight = totalHeight[localRow - 1][localCol];
		}
	}
	velPtr->T = fmax(0.0f, velPtr->T + precompMultiplier * ( thisHeight - neighbourHeight));
	//-----------------------
	// Bottom pipe
	//-----------------------
	if( i == (N.y-1) )
	{
		if(boundaries[0].reflect)
			neighbourHeight = thisHeight;
		else
			neighbourHeight = boundaries[0].level;
	}
	else
	{
		if( threadIdx.y == (blockDim.y-1) )
		{
			//water
			neighbourPtr = (float*)((char*)waterField + (i+1) * waterPitch) + j;
			neighbourHeight = *neighbourPtr;
			//sum up layers
			for(int f = 0; f < layerCount; ++f)
			{
				tmp = (float*)((char*)layers[f].ptr + (i+1) * layers[f].pitch) + j;
				neighbourHeight += *tmp;
			}
		}
		else
		{
			neighbourHeight = totalHeight[localRow + 1][localCol];
		}
	}
	velPtr->B = fmax(0.0f, velPtr->B + precompMultiplier * ( thisHeight - neighbourHeight));
	//-----------------------
	// Left pipe
	//-----------------------
	if( j == 0 )
	{
		if(boundaries[3].reflect)
			neighbourHeight = thisHeight;
		else
			neighbourHeight = boundaries[3].level;
	}
	else
	{
		if( threadIdx.x == 0 )
		{
		//water
			neighbourPtr = (float*)((char*)waterField + i * waterPitch) + (j-1) ;
			neighbourHeight = *neighbourPtr;
			//sum up layers
			for(int f = 0; f < layerCount; ++f)
			{
				tmp = (float*)((char*)layers[f].ptr + i * layers[f].pitch) + (j-1);
				neighbourHeight += *tmp;
			}
		}
		else
		{
			neighbourHeight = totalHeight[localRow][localCol - 1];
		}
	}
	velPtr->L = fmax(0.0f, velPtr->L + precompMultiplier * ( thisHeight - neighbourHeight));
	//-----------------------
	// Right pipe
	//-----------------------
	if( j == (N.x-1) )
	{
		if(boundaries[2].reflect)
			neighbourHeight = thisHeight;
		else
			neighbourHeight = boundaries[2].level;
	}
	else
	{
		if( threadIdx.x == (blockDim.x - 1) )
		{
			//water
			neighbourPtr = (float*)((char*)waterField + i * waterPitch) + (j+1);
			neighbourHeight = *neighbourPtr;
			//sum up layers
			for(int f = 0; f < layerCount; ++f)
			{
				tmp = (float*)((char*)layers[f].ptr + i * layers[f].pitch) + (j+1);
				neighbourHeight += *tmp;
			}
		}
		else
		{
			neighbourHeight = totalHeight[localRow][localCol + 1];
		}
	}
	velPtr->R = fmax(0.0f, velPtr->R + precompMultiplier * ( thisHeight - neighbourHeight));

	//-----------------------
	// Scale if the velocities would lead to an outflow greater than the cells water amount
	//-----------------------
	float sumOut = velPtr->B + velPtr->L + velPtr->R + velPtr->T;
	if(sumOut > *waterPtr)
	{
		float factor = *waterPtr/(sumOut);

		velPtr->B *= factor;
		velPtr->T *= factor;
		velPtr->R *= factor;
		velPtr->L *= factor;
	}
}

extern void cw_positiveVelocitiesBTRL( cell4Mem& lastVelocities, floatMem& lastWaterHeights,
							const std::vector <floatMem>& lastMaterialHeights, size_2D size,
							byteMem& activeBlocks, floatMem& totalHeightStorage)
{
	//convert material layer vector to array
	layer* linfo = new layer[lastMaterialHeights.size()];
	for(int i = 0; i < lastMaterialHeights.size(); ++i)
	{
		linfo[i].pitch = lastMaterialHeights[i].pitch;
		linfo[i].ptr = lastMaterialHeights[i].devPtr;
	}

	//stream layer info to device
	layer* linfoD;
	checkCudaErrors(cudaMalloc(&linfoD, lastMaterialHeights.size() * sizeof(layer)) );
	checkCudaErrors(cudaMemcpy(linfoD,linfo,    lastMaterialHeights.size() * sizeof(layer), 
		            cudaMemcpyHostToDevice));
	delete[] linfo;

	size_2D sizeInv(size.z,size.x);
	//run kernel
	positiveVelocitiesBTRL <<< getNumBlocks2D(sizeInv), getThreadsPerBlock2D() >>>
		(lastVelocities.devPtr, lastVelocities.pitch, 
		 lastWaterHeights.devPtr, lastWaterHeights.pitch, 
		 linfoD, lastMaterialHeights.size(), 
		 activeBlocks.devPtr, activeBlocks.pitch,
		 totalHeightStorage.devPtr, totalHeightStorage.pitch,
		 dim3FromSize_2D(sizeInv));
	checkCudaErrors(cudaGetLastError());

	//free layer info
	checkCudaErrors(cudaFree(linfoD));
}

//------------------------------------------------------------------------
// transport water using velocities
//------------------------------------------------------------------------

__global__ void computeWaterLevel( cell4* velField, size_t velPitch, 
								   float* waterField, size_t waterPitch,
								   bool* activeBlocks, size_t abPitch, 
								   float* soilField, size_t soilPitch,
								   bool erode, float erodeconst,
								   dim3 N)
{
	//execute block
	DEFINE_DEFAULT_KERNEL_WORKID_2D(j,i);
	RETURN_IF_OUTSIDE_2D(j,i,N);

	cell4* velPtr = (cell4*)((char*)velField + i * velPitch) + j;
	float* waterPtr = (float*)((char*)waterField + i * waterPitch) + j;
	float* soilPtr = (float*)((char*)soilField + i * soilPitch) + j;

	//------------------------
	// Transport water
	//------------------------

	float sumIn;
	float sumOut;

	sumOut = velPtr->B + velPtr->T + velPtr->R + velPtr->L;
	sumIn = 0.0f;
	
	cell4* neighbourPtr;
	//TOP NEIGHBOUR
	if( i!= 0)
	{
		 neighbourPtr = (cell4*)((char*)velField + (i-1) * velPitch) + j;
		 sumIn += neighbourPtr->B;
	}
	//BOTTOM NEIGHBOUR
	if( i!= (N.y-1))
	{
		 neighbourPtr = (cell4*)((char*)velField + (i+1) * velPitch) + j;
		 sumIn += neighbourPtr->T;
	}
	//LEFT NEIGHBOUR
	if( j!= 0)
	{
		 neighbourPtr = (cell4*)((char*)velField + i * velPitch) + (j-1);
		 sumIn += neighbourPtr->R;
	}
	//RIGHT NEIGHBOUR
	if( j!= (N.x-1))
	{
		 neighbourPtr = (cell4*)((char*)velField + i * velPitch) + (j+1);
		 sumIn += neighbourPtr->L;
	}
	float net = (sc[CONSTANT_TIMESTEP] * (sumIn-sumOut));
	*waterPtr += net;	
	if(erode)
		*soilPtr += erodeconst*net;
}

extern void cw_computeTransport( cell4Mem& velocities, floatMem& waterHeights, size_2D size,
								  byteMem& activeBlocks, floatMem& soil, bool erode, float erodeconst)
{
	size_2D sizeInv(size.z,size.x);
	computeWaterLevel <<< getNumBlocks2D(sizeInv), getThreadsPerBlock2D() >>>
		( velocities.devPtr, velocities.pitch, 
		  waterHeights.devPtr, waterHeights.pitch,
		  activeBlocks.devPtr, activeBlocks.pitch,
		  soil.devPtr,soil.pitch, erode, erodeconst,
		 dim3FromSize_2D(sizeInv));
	checkCudaErrors(cudaGetLastError());
}