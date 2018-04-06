#include "terrain.h"
#include "3rdParty\IL\il.h"
#include "util\global.h"

//------------------------------------------------------------------------
// External functions defined in terrain.cu
//------------------------------------------------------------------------

extern void cw_updateTerrainDataImage(const std::vector<terrainLayer>& layers, const floatMem& water, size_2D size, 
	                                     const byteMem& activeBlocks, bool forceUpdate, 
										 cudaGraphicsResource_t heightDest, cudaGraphicsResource_t matDest = 0 );

extern void cw_addToLayerFrom2DArray(const floatMem& repr, float *amounts, float amountScale, size_2D size);
extern void cw_memsetFloat2D( const floatMem repr, size_2D size, float value);

extern float* cw_writeLayersToArray( const floatMem& repr, size_2D size, bool normalize);
extern float* cw_writeLayersToArray( const std::vector < floatMem > & reprs, size_2D size,bool normalize);

extern void cw_gaussianFilter(cudaGraphicsResource_t heightResource, float sigma);
//------------------------------------------------------------------------
//------------------------------------------------------------------------

static floatMem initLayerFrom2DArray(float* heights, float heightScale , size_2D size)
{
	//alloc terrain memory
	floatMem mem;
	checkCudaErrors(  cudaMallocPitch(&mem.devPtr,&mem.pitch
					 ,size.cols * sizeof(float), size.rows));

	//set layer to 0
	cw_memsetFloat2D(mem, size, 0.0);
	//copy initial values
	cw_addToLayerFrom2DArray(mem,heights,heightScale, size);

	return mem;
}

//------------------------------------------------------------------------
// Terrain Work Handle
//------------------------------------------------------------------------

TerrainWorkHandle::TerrainWorkHandle(Terrain* base) 
	: myBase(base){}

//########################################################################
// Terrain definitions
//########################################################################

//------------------------------------------------------------------------
// General
//------------------------------------------------------------------------

Terrain::Terrain(size_2D terrainSize)
	: heightReceiver(0)
	, matReceiver(0)
	, size(terrainSize)
{
	pgcWater.devPtr = 0;

	//init water field
	checkCudaErrors(cudaMallocPitch(&pgcWater.devPtr,&pgcWater.pitch
		, size.cols * sizeof(float), size.rows));
	resetWater();

	//init blocks field
	dim3 computeGridDim = getNumBlocks2D(size);
	blockCount.rows = computeGridDim.x;
	blockCount.cols = computeGridDim.y;

	checkCudaErrors(cudaMallocPitch(&activeBlocks.devPtr, & activeBlocks.pitch,
		blockCount.cols, blockCount.height));
	//initially all blocks are active
	checkCudaErrors(cudaMemset2D(activeBlocks.devPtr, activeBlocks.pitch, 1 , 
					blockCount.cols, blockCount.rows));

}
Terrain::~Terrain()
{
	checkCudaErrors(cudaFree(pgcWater.devPtr));
	checkCudaErrors(cudaFree(activeBlocks.devPtr));

	for (auto iter = pgcMatHeight.begin(); iter != pgcMatHeight.end(); ++iter)
	{
		checkCudaErrors(cudaFree(iter->field.devPtr));
	}
}

TerrainWorkHandle* Terrain::getWorkHandle()
{
	return new TerrainWorkHandle(this);
}

void Terrain::resetWater()
{
	cw_memsetFloat2D(pgcWater,size,0.0);
}

void Terrain::removeLayer(unsigned int layerID)
{
	_removeLayer(layerID,true);
}

void Terrain::_removeLayer(unsigned int layerID, bool canClearWaterMem)
{
	assert(layerIDValid(layerID) && "Can not delete non-existant layer");

	int i = 0;
	for(auto iter = pgcMatHeight.begin(); iter != pgcMatHeight.end(); ++iter, ++i)
	{
		if( i == layerID)
		{
			iter->mat->decrementUseCount();
			checkCudaErrors(cudaFree(iter->field.devPtr));
			pgcMatHeight.erase(iter);
			return;
		}
	}
}

void Terrain::setLayerMaterial(unsigned int layerID, Material* material)
{
	assert(material && "Invalid Material Pointer");
	assert(layerIDValid(layerID) && "Can not access LayerID!");

	pgcMatHeight[layerID].mat->decrementUseCount();
	pgcMatHeight[layerID].mat = material;
	material->incrementUseCount();
}

//------------------------------------------------------------------------
// Input
//------------------------------------------------------------------------

void Terrain::addEmptyLayer( Material* material )
{
	float* heights = new float[size.rows * size.cols];

	for (size_t i = 0; i < size.rows ; ++i) 
	{
		for (size_t j = 0; j < size.cols; ++j)
		{
			heights[i*size.cols + j] = 0.0f;
		}
	}
	//for safety, even if 0.0 without above init could give us what we want, there may be NaNs or
	//other invalid values.
	addLayer(heights,size,0.0,material);
	delete[] heights;
}

void Terrain::addLayer(const float* source, size_2D s, float heightScale, Material* material)
{
	assert( source != 0 && "Invalid source pointer");
	assert(material && "Invalid Material Pointer");
	assert(size.x == s.x && size.z == s.z && "Terrain size mismatch!");
	
	terrainLayer l;
	l.field = initLayerFrom2DArray(const_cast<float*> (source), heightScale, size);

	l.mat = material;
	material->incrementUseCount();
	pgcMatHeight.push_back(l);
}



//------------------------------------------------------------------------
// Output
//------------------------------------------------------------------------


float* Terrain::writeLayer(size_2D& size, unsigned int layerID) const
{
	assert(layerIDValid(layerID) && "Can not write to non-existant layer!");
	size = this->size;
	return cw_writeLayersToArray(pgcMatHeight[layerID].field,size,false);
}

float* Terrain::write(size_2D& size, bool writeWater) const 
{
	assert(isInitialized()  && "Can not write empty Terrain");

	//Quick&dirty conversion
	std::vector <floatMem> tempVec;
	for(auto iter = pgcMatHeight.begin(); iter != pgcMatHeight.end(); iter++)
		tempVec.push_back(iter->field);

	tempVec.push_back(pgcWater);

	size = this->size;
	return cw_writeLayersToArray(tempVec,size,false);
}
float* Terrain::writeWater(size_2D& size) const
{
	size = this->size;
	return cw_writeLayersToArray(pgcWater,size,false);
}


//------------------------------------------------------------------------
// External updates
//------------------------------------------------------------------------

void Terrain::updateDataReceivers() const
{
	assert(isInitialized() && "Terrain empty");
	cw_updateTerrainDataImage(pgcMatHeight, pgcWater, size, activeBlocks, false,
		heightReceiver, matReceiver);
}

void Terrain::forceUpdateDataReceivers() const
{
	assert(isInitialized() && "Terrain empty");
	cw_updateTerrainDataImage(pgcMatHeight, pgcWater, size, activeBlocks, true,
		heightReceiver, matReceiver);
}

//------------------------------------------------------------------------
// Layer manipulation
//------------------------------------------------------------------------


void Terrain::collapseLayers(Material* material, bool collapseWater)
{
	assert(material && "Invalid Material Pointer");

	//Write a float map from all layers. Delete all layers. Init new layer0 with that map.
	std::vector <floatMem> tempVec;
	for(auto iter = pgcMatHeight.begin(); iter != pgcMatHeight.end(); iter++)
		tempVec.push_back(iter->field);

	if(collapseWater)
		tempVec.push_back(pgcWater);

	float* temp = cw_writeLayersToArray(tempVec,size, false);
	
	while( pgcMatHeight.size() != 0 )
		_removeLayer(0,false);

	terrainLayer l;
	l.field = initLayerFrom2DArray(temp,1.0,size);

	l.mat = material;
	material->incrementUseCount();

	pgcMatHeight.push_back(l);

	if(collapseWater)
	{
		resetWater();
	}
}

void Terrain::addLayerHeight(const float* source,const size_2D s, unsigned int layerID, float heightScale)
{
	assert(layerIDValid(layerID) && "Layer does not exist!");
	assert( source != 0 && "Invalid source pointer");
	assert(size.x == s.x && size.z == s.z && "Terrain size mismatch!");

	cw_addToLayerFrom2DArray(pgcMatHeight[layerID].field, const_cast<float*> (source), heightScale, size);
}

void Terrain::addWater(const float* source,const size_2D s, float heightScale)
{
	assert(isInitialized() && "Terrain empty");
	assert( source != 0 && "Invalid source pointer");
	assert(size.x == s.x && size.z == s.z && "Terrain size mismatch!");

	cw_addToLayerFrom2DArray(pgcWater, const_cast<float*> (source), heightScale, size);
}