#ifndef SIMULATOR_MANAGER_H
#define SIMULATOR_MANAGER_H

#include <vector>

#include "simulator.h"
#include "terrain.h"

class SimulationManager
{
public:
	SimulationManager(TerrainWorkHandle* base, SharedCudaResources* rcMan,
		const std::vector<Simulator*>& simulators);
	~SimulationManager();


	//execute simulators in order
	void update();
private:
	TerrainWorkHandle* handle;
	
	SharedCudaResources* rc;
	std::vector<Simulator*> simulators;
};


#endif //SIMULATOR_MANAGER_H