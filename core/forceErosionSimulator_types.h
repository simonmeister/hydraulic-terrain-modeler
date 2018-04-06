#ifndef FORCE_EROSION_SIMULATOR_TYPES_H
#define FORCE_EROSION_SIMULATOR_TYPES_H

#include "util\cuda_util.h"

struct cell2
{
	float x;
	float z;
};

typedef memCuda2D<cell2> cell2Mem;

#endif // FORCE_EROSION_SIMULATOR_TYPES_H