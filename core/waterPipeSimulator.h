#ifndef WATER_SIMULATOR_H
#define WATER_SIMULATOR_H

#include "simulator.h"
#include "terrain.h"
#include "util/global.h"
#include "WaterPipeSimulator_types.h"

const float DEFAULT_DRYTH = 0.001f;
const float DEFAULT_CONST = 0.3f;

const int DEFAULT_BOUNDCOND = 0;

//HACK.: "dryout" kernels are encapsulated inside of WPS - they
	//should really get a own simulator (to be consistent)
class WaterPipeSimulator : public Simulator
{
public:
	WaterPipeSimulator();
	virtual ~WaterPipeSimulator();



	void setTimestep(float step);

	void setDryTreshold(float th);

	inline void setDryOut(bool doit)
	{ dryOut = doit; }

	inline bool getDrysOut()
	{ return dryOut; }

	inline float getTimestep()
	{ return deltaT; }

	inline float getDryTreshold()
	{ return dryVal; }

	enum boundPosition
	{
		BOUND_BOTTOM = (1u << 0),
		BOUND_TOP    = (1u << 1),
		BOUND_RIGHT  = (1u << 2),
		BOUND_LEFT   = (1u << 3),
		BOUND_ALL    = (1u << 4)
	};

	typedef int boundpos;

	void setBoundaryReflect( boundpos pos );

	//is allowed to be negative, so a greater outflux may be achieved
	void setBoundaryFixed( boundpos pos, float level );

	inline void setActive(bool active)
	{this->active = active; }

	inline bool isActive() const 
	{ return active;}


	// copies everything resolution-independent to destination
	void copySettings(WaterPipeSimulator& from);
private:
	void initialize(SharedCudaResources* rc, TerrainWorkHandle* ter);
	void cleanUp();
	void step();

	SharedCudaResources* rc;
	TerrainWorkHandle* ter;

	bool active;
	float deltaT;
	float dryVal;
	bool dryOut;

	bool needsActiveUpdate;

	boundCond* bounds;
	cell4Mem velocitiesBTRL;
	floatMem lastTotalHeights;

	//Very small buffer.
	intMem auxActivities;
public:
	bool erode;
	float erodeconst;
};

#endif //WATER_SIMULATOR_H