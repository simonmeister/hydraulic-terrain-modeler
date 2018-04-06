#ifndef TERRAIN_STRUCTS_H
#define TERRAIN_STRUCTS_H

#include "util\cuda_util.h"
#include "material.h"

//TEMPORARY typedef
typedef unsigned int materialRef;

struct terrainLayer
{
	floatMem field;
	Material* mat;
};

struct linfo
{
	float* ptr;
	size_t pitch;

	unsigned int mat;
	float materialConstant;
};
#endif //TERRAIN_STRUCTS_H