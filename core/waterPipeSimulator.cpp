#include "WaterPipeSimulator.h"

#include <cuda_runtime.h>
#include "3rdParty\cuda-helper\helper_cuda.h"
#include "util\global.h"

extern void writePipeBounds(boundCond* b);
extern void cw_setConstant(WaterPipeSimulatorConst c, float value);

extern void cw_memsetCell4(float value, const cell4Mem& repr, size_2D size);

extern void cw_positiveVelocitiesBTRL( cell4Mem& lastVelocities, floatMem& lastWaterHeights,
							const std::vector<floatMem>& lastMaterialHeights, size_2D size,
							byteMem& activeBlocks, floatMem& totalHeightStorage);

extern void cw_computeTransport( cell4Mem& velocities, floatMem& waterHeights, size_2D size,
								  byteMem& activeBlocks, floatMem& soil, bool erode, float erodeconst);

extern void cw_blockActivity( floatMem& waterHeights, byteMem& activeBlocksResult, intMem& activeBlocksTemp
	                         , size_2D numAb , size_2D size);

extern void cw_blocksSetAllActive( byteMem& activeBlocks, size_2D size);

extern void cw_memsetFloat2D(floatMem field, size_2D size,  float value);


WaterPipeSimulator::WaterPipeSimulator()
: active(true)
, deltaT(DEFAULT_TIMESTEP)
, dryOut(false)
, erode(false)
, erodeconst(DEFAULT_CONST)
{
	velocitiesBTRL.devPtr = 0;
	auxActivities.devPtr = 0;
	lastTotalHeights.devPtr = 0;
	bounds = new boundCond[4];
	for(int i = 0; i<4;++i)
		bounds[i].reflect = true;
	writePipeBounds(bounds);

	cw_setConstant(CONSTANT_TIMESTEP,DEFAULT_TIMESTEP);
	cw_setConstant(CONSTANT_WATERDRY_THRESHOLD,DEFAULT_DRYTH);
}

WaterPipeSimulator::~WaterPipeSimulator()
{
	delete[] bounds;
}

void WaterPipeSimulator::copySettings(WaterPipeSimulator& from)
{
	active = from.active;
	deltaT = from.deltaT;
	dryVal = from.dryVal;
	erode = from.erode;
	erodeconst = from.erodeconst;
	memcpy(bounds, from.bounds, sizeof(boundCond)*4);

	writePipeBounds(bounds);
	cw_setConstant(CONSTANT_TIMESTEP,deltaT);
	cw_setConstant(CONSTANT_WATERDRY_THRESHOLD,dryVal);
}

void WaterPipeSimulator::setTimestep(float step)
{
	assert(step > 0.000000f && "Negative timestep");
	deltaT = step;
	cw_setConstant(CONSTANT_TIMESTEP,deltaT);
}

void WaterPipeSimulator::setDryTreshold(float th)
{
	assert(th > 0.000000f && "Negative treshold");
	dryVal = th;
	cw_setConstant(CONSTANT_WATERDRY_THRESHOLD,dryVal);
}

void WaterPipeSimulator::step()
{
	static int update = 0;
	assert(velocitiesBTRL.devPtr);
	//calculate which blocks are active
	if(update == 0)
	{
		if(dryOut)
		{
			cw_blockActivity( ter->water(), ter->activeBlocks(), auxActivities,
					ter->blockCount(), ter->size() );
		}
		else
		{
			cw_blocksSetAllActive(ter->activeBlocks(), ter->blockCount());
		}
		update = 16;
	}
	else
	{
		--update;
	}

	//update velocities
	cw_positiveVelocitiesBTRL( velocitiesBTRL, ter->water(), ter->layers() ,ter->size(), ter->activeBlocks(),lastTotalHeights);
	//update waterheight
	cw_computeTransport(velocitiesBTRL, ter->water(),  ter->size(), ter->activeBlocks(),ter->layer(0),erode,erodeconst);
}

void WaterPipeSimulator::initialize(SharedCudaResources* r, TerrainWorkHandle* t)
{
	rc = r;
	ter = t;
	//Alloc flow field
	checkCudaErrors(cudaMallocPitch(&velocitiesBTRL.devPtr,&velocitiesBTRL.pitch,
		ter->size().cols * sizeof(cell4), ter->size().rows));
	//set to 0
	cw_memsetCell4(0.0,velocitiesBTRL,ter->size());
	
	//alloc helper field
	checkCudaErrors(cudaMallocPitch(&auxActivities.devPtr, &auxActivities.pitch,
		ter->blockCount().cols, ter->blockCount().rows));

	//alloc TH helper field
	checkCudaErrors(cudaMallocPitch(&lastTotalHeights.devPtr, &lastTotalHeights.pitch,
		ter->size().cols * sizeof(cell4), ter->size().rows));
	//init
	cw_memsetFloat2D(lastTotalHeights,ter->size(),0.0);
	r->flowField = velocitiesBTRL;
	r->heights = lastTotalHeights;

}

void WaterPipeSimulator::setBoundaryReflect( boundpos pos )
{
	bool all = pos & BOUND_ALL;

	if( (pos & BOUND_BOTTOM) || all)
		bounds[0].reflect = true;
	if( (pos & BOUND_TOP) || all)
		bounds[1].reflect = true;
	if( (pos & BOUND_RIGHT) || all)
		bounds[2].reflect = true;
	if( (pos & BOUND_LEFT) || all)
		bounds[3].reflect = true;
	writePipeBounds(bounds);
}

void WaterPipeSimulator::setBoundaryFixed( boundpos pos, float level )
{
	bool all = pos & BOUND_ALL;

	if( (pos & BOUND_BOTTOM) || all)
	{
		bounds[0].reflect = false;
		bounds[0].level = level;
	}
	if( (pos & BOUND_TOP) || all)
	{
		bounds[1].reflect = false;
		bounds[1].level = level;
	}
	if( (pos & BOUND_RIGHT) || all)
	{
		bounds[2].reflect = false;
		bounds[2].level = level;
	}
	if( (pos & BOUND_LEFT) || all)
	{
		bounds[3].reflect = false;
		bounds[3].level = level;
	}

	writePipeBounds(bounds);
}

void WaterPipeSimulator::cleanUp()
{
	checkCudaErrors(cudaFree(auxActivities.devPtr));
	checkCudaErrors(cudaFree(velocitiesBTRL.devPtr));
	checkCudaErrors(cudaFree(lastTotalHeights.devPtr));
}