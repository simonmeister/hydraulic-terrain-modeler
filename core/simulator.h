#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "terrain.h"
#include "util\cuda_util.h"
#include "util\global.h"
#include "waterPipeSimulator_types.h"

#include <vector>
#include <unordered_map>

struct simResource
{
	floatMem field;
	size_2D size;
	bool permanent;
};

class SharedCudaResources
{
public:
	cell4Mem flowField;
	floatMem heights;
	/* Register a resource that may be used by all subsequent simulators. 
	If it is non permanent, it will be deleted after the current timestep.
	void register2DResource(const std::string& name, size_2D size, bool permanent);

	void unregister2DResource(const std::string& name);

	void clearNonPermanent();

	inline simResource getResource(const std::string& name)
	{ 
		assert(resources2D.find(name) != resources2D.end() &&
			 "Resource does not exist!");
		return resources2D[name]; 
	}


private:
	std::unordered_map < std::string, simResource> resources2D; */
};

class SimulationManager;

class Simulator
{
	friend class SimulationManager;

public:
	Simulator(){}
	virtual ~Simulator(){}

	

	virtual void setActive(bool active)= 0;
	virtual bool isActive() const = 0;

private:	
	/* These functions can only be called by SimulationManager */

	//Use initialize function to register resources etc.
	virtual void initialize(SharedCudaResources* rc, TerrainWorkHandle* ter) = 0;
	virtual void cleanUp() = 0;
	//Use step function to do work.
	virtual void step() = 0;
};


#endif //SIMULATOR_H