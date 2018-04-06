#include "forceErosionSimulator.h"

#include "waterPipeSimulator_types.h"
#include <vector>

extern void cw_memsetFloat2D(floatMem field, size_2D size,  float value);

extern void cw_computeSediment( floatMem& sedimentField, cell2Mem& velocityField, std::vector<floatMem> layers,
	floatMem& heights,  cell4Mem& flowField , byteMem& activeBlocks ,
	floatMem& water, floatMem& lastWater, size_2D size, float diss, float dep, std::vector<float> capacities, bool norm);

extern void cw_computeSedimentTransport(floatMem& sedimentField, floatMem& newValues, cell2Mem& velocityField, 
	byteMem& activeBlocks, size_2D size, float timeStep);

ForceErosionSimulator::ForceErosionSimulator()
: active(true)
, deltaT(DEFAULT_TIMESTEP)
, dep(0.1)
, diss(0.1)
, normalize(true)
{}

void ForceErosionSimulator::copySettings(ForceErosionSimulator& from)
{
	deltaT = from.deltaT;
	active = from.active;
	diss = from.diss;
	dep = from.dep;
}

void ForceErosionSimulator::step()
{
	//compute sediment amounts
	cw_computeSediment(sediment, velocities, ter->layers(), rc->heights, rc->flowField, ter->activeBlocks(),
		ter->water(), lastWater, ter->size(),diss,dep,ter->capacities(),normalize);
	//transport
	cw_computeSedimentTransport(sediment, sedimentTemp, velocities, ter->activeBlocks(), ter->size(), deltaT);
	//copy temp data
	checkCudaErrors(cudaMemcpy2D(sediment.devPtr,sediment.pitch,
		sedimentTemp.devPtr,sedimentTemp.pitch, sizeof(float) * ter->size().width, ter->size().height, cudaMemcpyDeviceToDevice));

	//obtain last height values (for next step)
	checkCudaErrors(cudaMemcpy2D(lastWater.devPtr, lastWater.pitch,
		ter->water().devPtr, ter->water().pitch, ter->size().z * sizeof(float), ter->size().x,
		cudaMemcpyDeviceToDevice));
}

void ForceErosionSimulator::initialize(SharedCudaResources* rc, TerrainWorkHandle* ter)
{
	this->rc = rc;
	this->ter = ter;

	//alloc and init last water
	checkCudaErrors(cudaMallocPitch(&lastWater.devPtr, &lastWater.pitch, 
		ter->size().width * sizeof(float), ter->size().height));
	cw_memsetFloat2D(lastWater,ter->size(),0.0f);

	//alloc and init sediment
	checkCudaErrors(cudaMallocPitch(&sediment.devPtr, &sediment.pitch, 
		ter->size().width * sizeof(float), ter->size().height));
	cw_memsetFloat2D(sediment,ter->size(),0.0f);
	
	checkCudaErrors(cudaMallocPitch(&sedimentTemp.devPtr, &sedimentTemp.pitch, 
		ter->size().width * sizeof(float), ter->size().height));

	//alloc velocities field
	checkCudaErrors(cudaMallocPitch(&velocities.devPtr, &velocities.pitch, sizeof(cell2) * ter->size().width,
		ter->size().height));
}

void ForceErosionSimulator::cleanUp()
{
	checkCudaErrors(cudaFree(lastWater.devPtr));
	checkCudaErrors(cudaFree(sediment.devPtr));
	checkCudaErrors(cudaFree(sedimentTemp.devPtr));
	checkCudaErrors(cudaFree(velocities.devPtr));
} 