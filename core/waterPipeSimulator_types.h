#ifndef WaterPipeSimulator_TYPES_H
#define WaterPipeSimulator_TYPES_H

#include "util\cuda_util.h"

struct cell4
{
	float L;
	float R;
	float T;
	float B;
};

typedef memCuda2D<cell4> cell4Mem;


struct layer
{
	float* ptr;
	size_t pitch;
	float cc;
};


struct boundCond
{
	float level;
	bool reflect;
};

enum WaterPipeSimulatorConst
{
	CONSTANT_WATERDRY_THRESHOLD = 0,
	CONSTANT_TIMESTEP = 1
};

#endif //WaterPipeSimulator_TYPES_H