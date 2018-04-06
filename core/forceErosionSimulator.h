#ifndef FORCE_EROSION_SIMULATOR
#define FORCE_EROSION_SIMULATOR

#include "simulator.h"
#include "terrain.h"
#include "forceErosionSimulator_types.h"


class ForceErosionSimulator : public Simulator
{
public:
	ForceErosionSimulator();
	virtual ~ForceErosionSimulator(){}

	inline void setTimestep(float step)
	{ deltaT = step;}

	inline float getTimestep()
	{ return deltaT;}

	inline float getDissolution()
	{ return  diss;}
		
	inline void setDissolution(float val)
	{ diss = val;}

	inline float getDeposition()
	{ return dep;}
		
	inline void setDeposition(float val)
	{ dep = val;}

	inline void setActive(bool active)
	{this->active = active; }

	inline bool isActive() const 
	{ return active;}

	inline bool normalizing() const
	{ return normalize; }

	inline void setNormalize(bool t)
	{ normalize = t; }


	// copies everything resolution-independent to destination
	void copySettings(ForceErosionSimulator& destination);
private:
	void initialize(SharedCudaResources* rc, TerrainWorkHandle* ter);
	void cleanUp();
	void step();

	SharedCudaResources* rc;
	TerrainWorkHandle* ter;

	bool active;
	bool normalize;
	float deltaT;

	float dep;
	float diss;


	floatMem lastWater;
	floatMem sediment;
	floatMem sedimentTemp;
	cell2Mem velocities;
};

#endif //FORCE_EROSION_SIMULATOR