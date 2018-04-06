#ifndef SOURCE_SIMULATOR_H
#define SOURCE_SIMULATOR_H

#include "simulator.h"
#include "terrain.h"
#include "sourcing_types.h"
//all positions are in normalized source space, so a rescale does not need 
//changes to the sources. X [0...1] Z [0...1]



struct source
{
	virtual ~source(){}
	float nPosX;
	float nPosZ;	
	//per cell intensity
	float intensity;
	sourceType tp;
	bool active;
};

struct radialSource : public source
{
	virtual ~radialSource(){}
	float nRadius;
};

struct rectSource : public source
{
	virtual ~rectSource(){}
	float nSideLenX;
	float nSideLenZ;
};

class SourceSimulator : public Simulator
{
public:	
	SourceSimulator();
	virtual ~SourceSimulator();


	inline void setActive(bool active)
	{this->active = active; }

	inline bool isActive() const 
	{ return active;}


	inline source* getSource(unsigned int idx)
	{assert(idx < sources.size());
	 return sources[idx]; precalculatedSources = false;}

	//of course, a negative intensity causes a sink effect instead of a source
	void addRadialSource(float nx, float nz, float intensity, float nrad);

	void addRectSource(float nx, float nz, float intensity, float nlenx, float nlenz);

	inline void addSource(source* src)
	{ sources.push_back(src); precalculatedSources = false;}

	void deleteSource(unsigned int idx);

	// copies everything resolution-independent to destination
	void copySettings(SourceSimulator& from);

	inline void renewSources()
	{ precalculatedSources = false; }

private:

	struct absoluteSource
	{
		virtual ~absoluteSource(){}
		size_2D from;
		size_2D to;
		float intensity;
		bool active;

		sourceType tp;
	};

	struct radialASource : public absoluteSource
	{
		virtual ~radialASource(){}
		size_2D pos;
		float rad;
	};

	//precalculate sources for speed
	void precalcAbsoluteSources();

	//
	void initialize(SharedCudaResources* rc, TerrainWorkHandle* ter);
	void cleanUp(){}
	void step();

	TerrainWorkHandle* ter;
	bool active;
	bool precalculatedSources;

	std::vector<source*> sources;
	std::vector<absoluteSource*> pcSources;
};

#endif //SOURCE_SIMULATOR_H