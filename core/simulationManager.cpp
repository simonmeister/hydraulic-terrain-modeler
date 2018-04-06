#include "simulationManager.h"
#include "terrain.h"


SimulationManager::SimulationManager(TerrainWorkHandle* b , SharedCudaResources* r,
	const std::vector<Simulator*>& s)
	: handle(b)
	, rc(r)
{
	simulators = s;
	for( auto iter = simulators.begin(); iter != simulators.end(); ++iter)
	{
		(*iter)->initialize(rc, handle);
	}
}

SimulationManager::~SimulationManager()
{
	for( auto iter = simulators.begin(); iter != simulators.end(); ++iter)
	{
		(*iter)->cleanUp();
	}
}

void SimulationManager::update()
{
	for( auto iter = simulators.begin(); iter != simulators.end(); ++iter)
	{
		if((*iter)->isActive())
			(*iter)->step();
	}
}