#include <cuda_runtime.h>
#include "forceErosionSimulator_types.h"
#include "util/cuda_util.h"
#include "util/global.h"
#include <vector>
#include "waterPipeSimulator_types.h"
#include "terrain_types.h"
#include "util\general_functions.cuh"


__device__ float clampZtN(float N, float val)
{//new range of val: [0, N]
	return __saturatef(val/N)* N;
}

__device__ float lerp(float a, float b, float w)
{ 
	return (a * w + (1.0f-w)* b);
}

//-----------------------------------------------------------------------------
// Compute Sediment kernel
//-----------------------------------------------------------------------------
__global__ void computeSediment(float* sedimentField, size_t sedimentPitch, 
	cell2* velocityField, size_t velocityPitch, 
	float* waterField, size_t waterPitch,
	float* lastWaterField, size_t lastWaterPitch,
	layer* layers, unsigned int layerCount, 
	float* lth, size_t lthPitch,
	cell4* flowField, size_t flowFieldPitch,
	bool* activeBlocks, size_t abPitch,
	dim3 N, float diss, float dep, bool norm)
{
	//early exit check
	bool* abPtr = (bool*)((char*)activeBlocks + blockIdx.y * abPitch) + blockIdx.x;
	if(!*abPtr)
		return;

	DEFINE_DEFAULT_KERNEL_WORKID_2D(j,i);
	RETURN_IF_OUTSIDE_2D(j,i,N);
	
	//-------------------------------------------------------------
	// compute velocity ( positive velocity in direction of growing indices )
	//-------------------------------------------------------------
	cell2* velPtr = (cell2*)((char*)velocityField + i * velocityPitch) + j;
	
	float* waterPtr  = (float*)((char*)waterField + i * waterPitch) + j;
	float* lastWaterPtr  = (float*)((char*)lastWaterField + i * lastWaterPitch) + j;
	float whAvg = (norm)? 0.5f * ( *waterPtr + *lastWaterPtr) : 1.0;

	float topFieldLowerOut, bottomFieldUpperOut, leftFieldRightOut, rightFieldLeftOut;
	//get neighbour flows to calculate velocity
	if( i == 0)
		topFieldLowerOut = 0.0f;
	else
		topFieldLowerOut = ((cell4*)((char*)flowField + (i-1) * flowFieldPitch) + j)->B;

	if( i == (N.y - 1))
		bottomFieldUpperOut = 0.0f;
	else
		bottomFieldUpperOut = ((cell4*)((char*)flowField + (i+1) * flowFieldPitch) + j)->T;

	if( j == 0)
		leftFieldRightOut = 0.0f;
	else
		leftFieldRightOut = ((cell4*)((char*)flowField + i * flowFieldPitch) + (j-1))->R;

	if( j == (N.x - 1))
		rightFieldLeftOut = 0.0f;
	else
		rightFieldLeftOut = ((cell4*)((char*)flowField + i * flowFieldPitch) + (j+1))->L;

	cell4* thisOflow = (cell4*)((char*)flowField + i * flowFieldPitch) + j;
	if(whAvg > 0.0)
	{
		velPtr->x = (topFieldLowerOut - thisOflow->T + thisOflow->B - bottomFieldUpperOut)/whAvg;
		velPtr->z = (leftFieldRightOut - thisOflow->L + thisOflow->R - rightFieldLeftOut)/whAvg;
	}
	else
	{
		velPtr->x = 0.0;
		velPtr->z = 0.0;
	}

	
	//-------------------------------------------------------------
	// compute capacity
	//-------------------------------------------------------------
	//compute tilt angle
	int idxTx,idxBx,idxRz,idxLz;
	//get neighbour height values
	idxRz = clampZtN(N.x-1,j+1);
	idxLz = clampZtN(N.x-1,j-1);
	idxTx = clampZtN(N.y-1,i-1);
	idxBx = clampZtN(N.y-1,i+1);
	float b,t,r,l,me;
	b = *((float*)((char*)lth + lthPitch * idxBx) + j);
	t = *((float*)((char*)lth + lthPitch * idxTx) + j);
	r = *((float*)((char*)lth + lthPitch * i) + idxRz);
	l = *((float*)((char*)lth + lthPitch * i) + idxLz);
	me = *((float*)((char*)lth + lthPitch * i) + j);

	b = fabsf(b-me)/sqrtf(powf(b-me,2));
	t = fabsf(t-me)/sqrtf(powf(t-me,2));
	r = fabsf(r-me)/sqrtf(powf(r-me,2));
	l = fabsf(l-me)/sqrtf(powf(l-me,2));
	float tilt = (b+t+r+l)/4.0;

	float cPrec = hypotf(velPtr->x,velPtr->z)*fmaxf(1.0f,0.1f);
	float cTop = cPrec* layers[layerCount-1].cc;
	//-------------------------------------------------------------
	// compute sediment level
	//-------------------------------------------------------------
	float* sedimentPtr = (float*)((char*)sedimentField + i * sedimentPitch) + j;
	float* layerPtr = (float*)((char*)layers[layerCount-1].ptr + i* layers[layerCount-1].pitch) + j;

	float st = *sedimentPtr;
	if(*sedimentPtr > cTop)
	{
		float m = dep * (st-cTop);
		m = (*sedimentPtr - m > 0.0 )? m : *sedimentPtr;
		*sedimentPtr -= m;
		*layerPtr += m;
	}
	else
	{
		float dTop = 0.0;
		float ca;
		for(int f = layerCount-1; f > -1; --f)
		{
			// replace ctop by c = cPrec* capacity[f];
			float c = cPrec * layers[f].cc;
			ca = c - dTop;
			if(ca > *sedimentPtr)
			{
				float m = diss * (c-st);
				m = (*layerPtr - m > 0.0 )? m : *layerPtr;
				*sedimentPtr += m;
				layerPtr = (float*)((char*)layers[f].ptr + i* layers[f].pitch) + j;
				*layerPtr -= m;
			}
			else
			{
				break;
			}
			dTop += *layerPtr;
		}
	}
}
extern void cw_computeSediment( floatMem& sedimentField, cell2Mem& velocityField, std::vector<floatMem> layers,
	floatMem& heights,  cell4Mem& flowField , byteMem& activeBlocks ,
	floatMem& water, floatMem& lastWater, size_2D size, float diss, float dep, std::vector<float> capacities, bool norm)
{
	//convert material layer vector to array
	layer* linfo = new layer[layers.size()];
	for(int i = 0; i < layers.size(); ++i)
	{
		linfo[i].pitch = layers[i].pitch;
		linfo[i].ptr = layers[i].devPtr;
		linfo[i].cc = capacities[i];
	}

	//stream layer info to device
	layer* linfoD;
	checkCudaErrors(cudaMalloc(&linfoD, layers.size() * sizeof(layer)) );
	checkCudaErrors(cudaMemcpy(linfoD,linfo,    layers.size() * sizeof(layer), 
		            cudaMemcpyHostToDevice));
	delete[] linfo;

	size_2D sizeInv(size.z,size.x);
	computeSediment <<< getNumBlocks2D(sizeInv), getThreadsPerBlock2D() >>>
		(sedimentField.devPtr, sedimentField.pitch, 
		velocityField.devPtr, velocityField.pitch, 
		water.devPtr, water.pitch,
		lastWater.devPtr, lastWater.pitch,
		linfoD, layers.size(), 
		heights.devPtr,heights.pitch,
		flowField.devPtr, flowField.pitch, 
		activeBlocks.devPtr, activeBlocks.pitch 
		, dim3FromSize_2D(sizeInv), diss, dep,norm);
	checkCudaErrors(cudaGetLastError());

	//free layer info
	checkCudaErrors(cudaFree(linfoD));
}

//-----------------------------------------------------------------------------
// Compute Sediment Transport kernel
//-----------------------------------------------------------------------------

__global__ void computeSedimentTransport(
	float* sedimentField,  size_t sedimentPitch, 
	float* sedimentWrite,  size_t swPitch, 
	cell2* velocityField, size_t velocityPitch, 
	bool* activeBlocks, size_t abPitch,
	float timeStep, dim3 N)
{
	//early exit check
	bool* abPtr = (bool*)((char*)activeBlocks + blockIdx.y * abPitch) + blockIdx.x;
	if(!*abPtr)
		return;

	DEFINE_DEFAULT_KERNEL_WORKID_2D(j,i);
	RETURN_IF_OUTSIDE_2D(j,i,N);

	//compute point using semi-lagrangian backtracing
	//current position is cell-centroid (idx+0.5)

	cell2* velPtr = (cell2*)((char*)velocityField + i * velocityPitch) + j;
	float velx = velPtr->x;
	float velz = velPtr->z;

	float tpx = (float)i+0.5 - velx * timeStep;
	float tpz = (float)j+0.5 - velz * timeStep;

	//nearest cell corner of point
	int npx = floorf(tpx + 0.5);
	int npz = floorf(tpz + 0.5);

	int idxTx,idxBx,idxRz,idxLz;
	//compute indices of surrounding cells
	idxRz = npz;
	idxLz = npz - 1;
	idxTx = npx - 1;
	idxBx = npx;

	//compute weights
	float wZR = tpz - 0.5 - ( (float)npz - 1.0);
	float wXB = tpx - 0.5 - ( (float)npx - 1.0);

	//points must be on the grid (the following method introduces "no-slip" boundary conditions.)
	idxRz = clampZtN(N.x-1,idxRz);
	idxLz = clampZtN(N.x-1,idxLz);
	idxTx = clampZtN(N.y-1,idxTx);
	idxBx = clampZtN(N.y-1,idxBx);

	//get values at the 4 surrounding points
	float tr,tl,br,bl;
	tr = *((float*)((char*)sedimentField + idxTx * sedimentPitch) + idxRz);
	tl = *((float*)((char*)sedimentField + idxTx * sedimentPitch) + idxLz);
	br = *((float*)((char*)sedimentField + idxBx * sedimentPitch) + idxRz);
	bl = *((float*)((char*)sedimentField + idxBx * sedimentPitch) + idxLz);

	//weighted (bilinear) interpolation
	float* thisCell = (float*)((char*)sedimentWrite + i * swPitch) + j;
	*thisCell =  lerp(lerp(br,bl, wZR),lerp(tr,tl, wZR),wXB);
}

extern void cw_computeSedimentTransport(floatMem& sedimentField, floatMem& newValues, cell2Mem& velocityField, 
	byteMem& activeBlocks, size_2D size, float timeStep)
{
	size_2D sizeInv(size.z,size.x);
	computeSedimentTransport <<< getNumBlocks2D(sizeInv) , getThreadsPerBlock2D() >>>
		(sedimentField.devPtr, sedimentField.pitch, 
		newValues.devPtr, newValues.pitch, 
		velocityField.devPtr, velocityField.pitch,
		activeBlocks.devPtr, activeBlocks.pitch,
		timeStep ,dim3FromSize_2D(sizeInv));
	checkCudaErrors(cudaGetLastError());
}