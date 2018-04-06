#include "sourceSimulator.h"
#include "util\global.h"
#include <cmath>

//fromIdx must not be greater than toIdx. toId must be smaller than gridSize.x or .z;
//a square gird is executed. if a thread is within the given radius of posId, it spawns water.
extern void cw_radialSourcing(floatMem& heights, size_2D gridSize,
	                        float perCellAmount, 
							size_2D fromIdx, size_2D toIdx, size_2D posIdx,
							float radius, float hardness, bool useFalloff);

//fromIdxX must not be greater than toIdxX. toIdX must be smaller than gridSize.x;
//The same applies to the Z dimension.
extern void cw_rectSourcing(floatMem& water, size_2D gridSize,
	                        float perCellAmount, 
							size_2D fromIdx, size_2D toIdx); 

SourceSimulator::SourceSimulator()
	: active(true)
	, precalculatedSources(false)
{}

void SourceSimulator::copySettings(SourceSimulator& from)
{
	for(auto iter = from.sources.begin(); iter != from.sources.end(); ++iter)
	{
		if((*iter)->tp == SRC_RADIAL)
			sources.push_back((source*) new radialSource(*((radialSource*)*iter)));
		else
			sources.push_back((source*) new rectSource(*((rectSource*)*iter)));

	}
	active = true;
	precalculatedSources = false;
}

void SourceSimulator::deleteSource(unsigned int idx)
{
	int i = 0;
	for(auto iter = sources.begin() ; i< sources.size(); ++i, ++iter)
	{
		if( i == idx)
		{
			sources.erase(iter);
			break;
		}
	}
	precalculatedSources = false;
}

void SourceSimulator::initialize(SharedCudaResources* rc, TerrainWorkHandle* ter)
{
	this->ter = ter;
}

SourceSimulator::~SourceSimulator()
{
	for(auto iter = sources.begin(); iter != sources.end(); ++iter)
	{
		delete *iter;
	}
	for(auto iter = pcSources.begin(); iter != pcSources.end(); ++iter)
	{
		delete *iter;
	}
}

void SourceSimulator::addRadialSource(float nx, float nz, float intensity, float nrad)
{
	assert(nx <= 1.0f && nx >= 0.0f && "Out of range");
	assert(nz <= 1.0f && nz >= 0.0f && "Out of range");

	radialSource* src = new radialSource;
	src->intensity = intensity;
	src->nPosX = nx;
	src->nPosZ = nz;
	src->nRadius = nrad;
	src->tp = SRC_RADIAL;
	src->active = true;

	sources.push_back((source*)src);

	//force recalc
	precalculatedSources = false;
}

void SourceSimulator::addRectSource(float nx, float nz, float intensity, float nlenx, float nlenz)
{
	assert(nx <= 1.0f && nx >= 0.0f && "Out of range");
	assert(nz <= 1.0f && nz >= 0.0f && "Out of range");

	rectSource* src = new rectSource;
	src->intensity = intensity;
	src->nPosX = nx;
	src->nPosZ = nz;
	src->nSideLenX = nlenx;
	src->nSideLenZ = nlenz;
	src->tp = SRC_RECTANGLE;
	src->active = true;

	sources.push_back((source*)src);

	//force recalc
	precalculatedSources = false;
}

void SourceSimulator::precalcAbsoluteSources()
{
	for(auto iter = pcSources.begin(); iter != pcSources.end(); ++iter)
	{
		delete *iter;
	}
	pcSources.clear();
	pcSources.shrink_to_fit();

	for(auto iter = sources.begin(); iter != sources.end(); ++iter)
	{
		//Invalid values?
		assert((*iter)->nPosX <= 1.0 && (*iter)->nPosX >= 0.0);
		assert((*iter)->nPosZ <= 1.0 && (*iter)->nPosZ >= 0.0);

		if((*iter)->tp == SRC_RECTANGLE)
		{
			rectSource* s = (rectSource*)(*iter);

			//calculate execution domain in normalized source coordinates
			float nxfrom,nzfrom,nxto, nzto;
			nxfrom = std::max<float>(0.0, s->nPosX - s->nSideLenX/2.0f);
			nzfrom = std::max<float>(0.0, s->nPosZ - s->nSideLenZ/2.0f);
			nxto =	 std::min<float>(1.0, s->nPosX + s->nSideLenX/2.0f);
			nzto =	 std::min<float>(1.0, s->nPosZ + s->nSideLenZ/2.0f);

			//normalized source coordinates to grid coordinates
			size_t xfrom,xto,zfrom,zto;

			xfrom =	(size_t)	(nxfrom * (ter->size().x-1));
			xto =	(size_t)	(nxto * (ter->size().x-1));
			zfrom =	(size_t)	(nzfrom * (ter->size().z-1));
			zto =	(size_t)	(nzto * (ter->size().z-1));

			//store
			absoluteSource* src = new absoluteSource();
			src->from = size_2D(xfrom,zfrom);
			src->to = size_2D(xto,zto);
			src->intensity = s->intensity;
			src->active = s->active;
			src->tp = SRC_RECTANGLE;
			pcSources.push_back(src);
		}
		else
		{
			radialSource* s = (radialSource*)(*iter);

			//calculate execution domain in normalized source coordinates
			float nxfrom,nzfrom,nxto, nzto;
			float rx,rz;
			//scale down the along larger dimension, as it will be scaled up to strong by the
			//second level of calculations
			if( ter->size().x >  ter->size().z)
			{
				float aspectRatio = ter->size().x / (float)ter->size().z;
				rz = s->nRadius;
				rx = s->nRadius/aspectRatio;
			}
			else
			{
				float aspectRatio = ter->size().z / (float)ter->size().x;
				rx = s->nRadius;
				rz = s->nRadius/aspectRatio;
			}

			nxfrom = std::max<float>(0.0, s->nPosX - rx);
			nzfrom = std::max<float>(0.0, s->nPosZ - rz);
			nxto =	 std::min<float>(1.0, s->nPosX + rx);
			nzto =	 std::min<float>(1.0, s->nPosZ + rz);

			//normalized source coordinates to grid coordinates
			size_t xfrom,xto,zfrom,zto, xpos, zpos;
			

			xfrom = (size_t)	(nxfrom * (ter->size().x-1));
			xto =	(size_t)	(nxto * (ter->size().x-1));
			zfrom =	(size_t)	(nzfrom * (ter->size().z-1));
			zto =	(size_t)	(nzto * (ter->size().z-1));

			xpos =	(size_t)	(s->nPosX  * (ter->size().x-1));
			zpos =	(size_t)	(s->nPosZ  * (ter->size().z-1));

			//store
			radialASource* src = new radialASource();
			src->from = size_2D(xfrom,zfrom);
			src->to = size_2D(xto,zto);
			src->intensity = s->intensity;
			src->active = s->active;
			src->rad = s->nRadius;
			src->tp = SRC_RADIAL;
			src->pos = size_2D(xpos,zpos);
			pcSources.push_back(src);
		}
	}
	precalculatedSources = true;
}

void SourceSimulator::step()
{
	if(!precalculatedSources)
		precalcAbsoluteSources();

	for(auto iter = pcSources.begin(); iter != pcSources.end(); ++iter)
	{
		if(!(*iter)->active)
			return;

		if((*iter)->tp == SRC_RECTANGLE)
		{
			absoluteSource* src = (*iter);
			//invoke kernel
			cw_rectSourcing(ter->water(), ter->size(), src->intensity,
				src->from,src->to);
		}
		else
		{
			radialASource* src = (radialASource*)(*iter);
			//invoke kernel
			cw_radialSourcing(ter->water(), ter->size(),src->intensity,
				src->from,src->to,src->pos,src->rad,1.0,false);
		}
	}
}