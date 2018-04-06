#ifndef TERRAIN_H
#define TERRAIN_H


#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "util/global.h"
#include "util\cuda_util.h"

#include "material.h"
#include "terrain_types.h"

//Forward decl. for friend 
class TerrainWorkHandle;

class Terrain
{

	friend class TerrainWorkHandle;

public:

	Terrain(size_2D terrainSize);
	virtual ~Terrain();


	TerrainWorkHandle* getWorkHandle();

	inline size_2D getSize() const 
	{ return size; }

	inline float getAspectRatio() const
	{ return (float)size.x/(float)size.z;}

	//Get count of material layers, exclude water
	inline unsigned int getLayerCount() const
	{ return pgcMatHeight.size(); }

	inline bool isSingleLayered() const
	{ return pgcMatHeight.size() == 1; }

	//To be initialized and usable for simulation there must be at least 1 layer
	inline bool isInitialized() const
	{ return pgcMatHeight.size() > 0; }




	/* These methods give access to updates for frequent rendering */
	inline void connectDataReceivers(cudaGraphicsResource_t heightR, cudaGraphicsResource_t materialR = 0)
	{ heightReceiver = heightR; matReceiver = materialR; }

	void disconnectDataReceivers() 
	{ heightReceiver = 0; matReceiver = 0;}

	//Updates connected resource with current terrain data
	void updateDataReceivers() const;
	void forceUpdateDataReceivers() const;


	//Set waterheights to 0 
	void resetWater();

	//Initialize one base layer with height 0
	void addEmptyLayer( Material* material);

	//Add this to the top of the layer-stack. If this is the first layer, init state goes from false to true.
	void addLayer(const float* source, size_2D size, float heightScale, Material* material);

	//Write all layers collapsed into a single image. Extension autodetection.
	float* write(size_2D& size, bool writeWater = false) const;

	float* writeLayer(size_2D& size, unsigned int layerID) const;

	float* writeWater(size_2D& size) const;

	void setLayerMaterial(unsigned int layerID, Material* material);

	void removeLayer(unsigned int layerID);

	//Collapse all layers into one. Useless when isSingleLayered is true.
	void collapseLayers(Material* material, bool collapseWater = false);

	void addLayerHeight(const float* source,const size_2D size, unsigned int layerID, float heightScale);

	void addWater(const float* source,const size_2D size, float heightScale);

protected:

	//in collapseLayers we want to delete all layers without a complete reset to uninitialized state
	void _removeLayer(unsigned int layerID, bool canClearWaterMem);

	inline bool layerIDValid(unsigned int id) const
	{ return id < pgcMatHeight.size(); }

	void initTerrain();

	size_2D size;
	size_2D blockCount;
	//The heightmap per-grid-cell values of size "size".
	std::vector< terrainLayer > pgcMatHeight;
	floatMem pgcWater;

	cudaGraphicsResource_t heightReceiver;
	cudaGraphicsResource_t matReceiver;

	//number of active cuda blocks
	byteMem activeBlocks;
};


class TerrainWorkHandle
{
public:
	TerrainWorkHandle(Terrain* base);
	virtual ~TerrainWorkHandle(){}

	inline floatMem water()
	{ return myBase->pgcWater; }

	/////////// REPLACE THIS TO SPEED UP 
	inline std::vector<floatMem> layers()
	{
		std::vector<floatMem> temp;
		for( auto iter = myBase->pgcMatHeight.begin(); iter !=  myBase->pgcMatHeight.end(); ++iter)
		{
			temp.push_back(iter->field);
		}
		return temp;
	}

	inline std::vector<float> capacities()
	{
		std::vector<float> temp;
		for( auto iter = myBase->pgcMatHeight.begin(); iter !=  myBase->pgcMatHeight.end(); ++iter)
		{
			temp.push_back(iter->mat->getDensity());
		}
		return temp;
	}

	inline floatMem layer(unsigned int id)
	{
		assert(myBase->layerIDValid(id));	
		return myBase->pgcMatHeight[id].field;
	}

	inline size_2D size() 
	{ return myBase->size; }

	inline size_2D blockCount()
	{ return myBase->blockCount;}

	inline byteMem activeBlocks()
	{ return myBase->activeBlocks;}
private:

	Terrain* myBase;
};

#endif //TERRAIN_H 