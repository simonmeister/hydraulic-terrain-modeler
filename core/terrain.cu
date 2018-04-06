

#include "terrain_types.h"
#include "util\global.h"

#include <cuda_runtime.h>
#include <vector>
#include "3rdParty\cuda-helper\helper_cuda.h"
#include "util\cuda_util.h"


#include "util/general_functions.cuh"


//------------------------------------------------------------------------
// Update Textures for rendering
//------------------------------------------------------------------------
surface<void, 2> surfHeight;
surface<void, 2> surfMat;

__global__ void updateDataImage( linfo* layers, size_t layerCount ,float* water,size_t waterPitch, 
								 bool* activeBlocks, size_t abPitch, bool forceUpdate,
	                             bool writeMaterial, dim3 N)
{
	//only update if something happens here or it is forced for a complete rerender)
	bool* abPtr = (bool*)((char*)activeBlocks + blockIdx.y * abPitch) + blockIdx.x;
	if(*abPtr || forceUpdate)
	{
		DEFINE_DEFAULT_KERNEL_WORKID_2D(j,i);
		RETURN_IF_OUTSIDE_2D(j,i,N);

		float* wPtr = (float*)((char*)water + i * waterPitch) + j;
		float matHeight = 0.0;
		unsigned char matID;
		//add up layers, assign material of highest non-zero layer
		for(int f = 0; f < layerCount; ++f)
		{
			float* lPtr = (float*)((char*)layers[f].ptr + i * layers[f].pitch) + j;
			matHeight += *lPtr;
			//ensure that a material-id is assigned by always assigning the lowest layer first
			if(*lPtr > 0.00 || f == 0)
				matID = layers[f].mat;
		}
		surf2Dwrite( make_float2(matHeight, *wPtr) , surfHeight, i*sizeof(float2), j);
		if(writeMaterial)
			surf2Dwrite( matID, surfMat, i*sizeof(unsigned char), j);
	}
}

extern void cw_updateTerrainDataImage(const std::vector<terrainLayer>& layers, const floatMem& water,size_2D size,
	                                   const byteMem& activeBlocks, bool forceUpdate,
	                                   cudaGraphicsResource_t heightDest, cudaGraphicsResource_t matDest = 0 )
{
	cudaArray* heightArray;
	checkCudaErrors(cudaGraphicsMapResources(1,&heightDest,0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&heightArray,heightDest,0,0));

	cudaArray* matArray;
	if(matDest)
	{	
		checkCudaErrors(cudaGraphicsMapResources(1,&matDest,0));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&matArray,matDest,0,0));
	}
	//mapped ####################################################################
	
	//copy layerinfo (pointer to layer, material id) to cuda mem
	linfo* layerinfo = new linfo[layers.size()];
	for(size_t i = 0; i < layers.size(); ++i)
	{
		layerinfo[i].mat = layers[i].mat->getID(); 
		layerinfo[i].pitch = layers[i].field.pitch;
		layerinfo[i].ptr = layers[i].field.devPtr;
	}
	linfo* layerinfoD;
	checkCudaErrors(cudaMalloc(&layerinfoD, layers.size() * sizeof(linfo)));
	checkCudaErrors(cudaMemcpy(layerinfoD,layerinfo,sizeof(linfo) * layers.size(),cudaMemcpyHostToDevice));
	delete[] layerinfo;

	
	cudaChannelFormatDesc desc;
	checkCudaErrors(cudaGetChannelDesc(&desc,heightArray));
	checkCudaErrors(cudaBindSurfaceToArray(surfHeight,heightArray));
	if(matDest)
	{
		checkCudaErrors(cudaGetChannelDesc(&desc,matArray));
		checkCudaErrors(cudaBindSurfaceToArray(surfMat,matArray));
	}

	size_2D sizeInv(size.z,size.x);
	/* Write water and layer height values */
	updateDataImage <<< getNumBlocks2D(sizeInv),getThreadsPerBlock2D() >>> 
		(layerinfoD, layers.size(),water.devPtr, water.pitch, 
		 activeBlocks.devPtr, activeBlocks.pitch, forceUpdate,
		matDest != 0, dim3FromSize_2D(sizeInv));
	checkCudaErrors(cudaGetLastError());
	
	//unmap #####################################################################
	checkCudaErrors(cudaGraphicsUnmapResources(1,&heightDest)); 
	if(matDest)
		checkCudaErrors(cudaGraphicsUnmapResources(1,&matDest)); 
	checkCudaErrors(cudaFree(layerinfoD));
}




//------------------------------------------------------------------------
//------------------------------------------------------------------------

extern float* cw_writeLayersToArray( const std::vector < floatMem > & reprs, size_2D size, bool normalize)
{
	float max;
	size_t fieldCellsCount = size.rows * size.cols;
	if(normalize)
	{
		/* First, compute the max value to divide by it later */
		//stream all layers to ram	
		float** temp = new float*[reprs.size()];
		for( size_t i = 0; i < reprs.size(); ++i)
		{
			temp[i] = new float[fieldCellsCount];

			checkCudaErrors(cudaMemcpy2D(temp[i],size.width* sizeof(float)
			,reprs[i].devPtr,reprs[i].pitch, 
			size.width * sizeof(float),size.height, cudaMemcpyDeviceToHost));
		}
		max = 0.0; // Height must be positive
		for(size_t i = 0; i < size.rows; ++i)
		{
			for( size_t j = 0; j < size.cols; ++j)
			{
				//compute value for this position by adding up layers
				float value = 0.0;
				for( size_t f = 0; f< reprs.size(); ++f)
				{
					value += temp[f][i* size.cols + j];
				}
				if( value > max )
					max = value;
			}
		}
		for( size_t i = 0; i < reprs.size(); ++i)
		{
			delete[] temp[i];
		}
		delete[] temp;
	}

	/* Now, add up all layers and optionally scale the result by max*/
	float* destD;
	size_t destPitch;
	checkCudaErrors(cudaMallocPitch(&destD, &destPitch, size.cols* sizeof(float),size.rows));
	memsetFloat2D <<< getNumBlocks2D(size), getThreadsPerBlock2D() >>>
		(destD, destPitch, 0.0f, dim3FromSize_2D(size) );
	for(auto iter = reprs.begin(); iter != reprs.end(); iter++)
	{
		increment2D_2D <<< getNumBlocks2D(size), getThreadsPerBlock2D() >>>
			(destD, destPitch, iter->devPtr, iter->pitch, dim3FromSize_2D(size));
		checkCudaErrors(cudaGetLastError());
	}

	if(normalize)
	{
		multiply2DScalar <<< getNumBlocks2D(size), getThreadsPerBlock2D() >>>
							 (1.0f/max, destD, destPitch, dim3FromSize_2D(size));
		checkCudaErrors(cudaGetLastError());
	}

	/* Copy this back to ram and return*/
	float* ret = new float[fieldCellsCount];
	checkCudaErrors(cudaMemcpy2D(ret, sizeof(float)* size.cols, 
		destD, destPitch, sizeof(float)* size.cols, size.rows, 
		cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(destD));
	return ret;
}

extern float* cw_writeLayersToArray( const floatMem& repr , size_2D size, bool normalize )
{
	std::vector< floatMem> vec;
	vec.push_back(repr);
	return cw_writeLayersToArray(vec,size,normalize);
}



//------------------------------------------------------------------------
//------------------------------------------------------------------------

extern void cw_addToLayerFrom2DArray(const floatMem& repr, float *amounts, float amountScale, size_2D size)
{
	//Alloc and stream increment field
	float* amountsPtr;
	size_t amountsPitch;
	checkCudaErrors(  cudaMallocPitch(&amountsPtr,&amountsPitch
					 ,size.cols * sizeof(float), size.rows));

	checkCudaErrors( cudaMemcpy2D(amountsPtr,amountsPitch,amounts , 
					 sizeof(float) * size.cols, sizeof(float) * size.cols, size.rows,
					 cudaMemcpyHostToDevice));

	//scale amounts 
	multiply2DScalar <<< getNumBlocks2D(size), getThreadsPerBlock2D() >>>
		( amountScale, amountsPtr, amountsPitch, dim3FromSize_2D(size));
	checkCudaErrors(cudaGetLastError());

	//increment repr with amounts
	increment2D_2D <<< getNumBlocks2D(size), getThreadsPerBlock2D() >>>
		(repr.devPtr, repr.pitch, amountsPtr, amountsPitch, dim3FromSize_2D(size));
	checkCudaErrors(cudaGetLastError());

	//free resources
	checkCudaErrors( cudaFree(amountsPtr));
}
