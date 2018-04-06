
// HyTM includes
#include "meshgen.h"
#include "util/global.h"
#include <cmath>
#include <algorithm>

////////////////////////////////////////////////////////////////
/* IndexedVertices definitions*/
IndexedVertices::IndexedVertices()
	: vertexCount(0)
	, elementCount(0)
	, vertices(nullptr)
	, elements(nullptr)
{}

void IndexedVertices::clear(IndexedVertices& t)
{
	SAFE_DELETE_ARRAY(t.vertices);
	SAFE_DELETE_ARRAY(t.elements);
}
////////////////////////////////////////////////////////////////
/* Vertices definitions */
Vertices::Vertices()
	: vertexCount(0)
	, vertices(nullptr)
{}

void Vertices::clear(Vertices& t)
{
	SAFE_DELETE_ARRAY(t.vertices);
}
//////////////////////////////////////////////////////////////
IndexedVertices quadPlane(size_t subdX, size_t subdZ,
	float lenX, float lenZ, bool sym)
{	
	size_t vertXDim = subdX + 1;
	size_t vertZDim = subdZ + 1;
	size_t vertCount = vertXDim * vertZDim;
	#define VERTS_PER_QUAD 4
	size_t elmCount = (subdX*subdZ)*VERTS_PER_QUAD;	

	vertex3f *vertexArray = new vertex3f[vertCount]; 
	unsigned int *elementArray = new unsigned int[elmCount];

	float posX;
	float posZ;
	if(sym)
	{
		posX = -lenX/2.0f;
		posZ = -lenZ/2.0f;
	}
	else
	{
		posX = 0.0;
		posZ = 0.0;
	}
	float stepX = lenX/(float)subdX;
	float stepZ = lenZ/(float)subdZ;

	/* build vertex array */
	for( size_t i = 0, offset = 0; i < vertXDim;
		++i, posX += stepX)
	{
		posZ = -lenZ/2.0f;
		for( size_t j = 0; j < vertZDim;
			++j, posZ += stepZ, ++offset)
		{
			vertexArray[offset].x = posX;
			vertexArray[offset].y = 0.0f;
			vertexArray[offset].z = posZ;
		}
	}

	/* build element array */
	for( size_t i = 0, offset = 0, vIndexTopLeft = 0; i < subdX; ++i, ++vIndexTopLeft)
	{
		for( size_t j = 0; j < subdZ;
			++j, offset += VERTS_PER_QUAD, ++vIndexTopLeft)
		{
			elementArray[offset]   = vIndexTopLeft;
			elementArray[offset+1] = vIndexTopLeft + 1;
			elementArray[offset+2] = vIndexTopLeft + subdZ + 2;
			elementArray[offset+3] = vIndexTopLeft + subdZ + 1;
		}
	}

	IndexedVertices result;
	result.elementCount = elmCount;
	result.vertexCount = vertCount;
	result.elements = elementArray;
	result.vertices = vertexArray;
	return result;
}
//////////////////////////////////////////////////////////////////////////////////
IndexedVertices trianglePlane(size_t subdX, size_t subdZ,
	float lenX, float lenZ, bool sym)
{
	size_t vertXDim = subdX + 1;
	size_t vertZDim = subdZ + 1;
	size_t vertCount = vertXDim * vertZDim;
	#define VERTS_PER_TRI 3
	#define TRIS_PER_QUAD 2
	size_t elmCount = (subdX * subdZ) * (VERTS_PER_TRI * TRIS_PER_QUAD);	

	vertex3f *vertexArray = new vertex3f[vertCount]; 
	unsigned int *elementArray = new unsigned int[elmCount];

	float posX;
	float posZ;
	if(sym)
	{
		posX = -lenX/2.0f;
		posZ = -lenZ/2.0f;
	}
	else
	{
		posX = 0.0;
		posZ = 0.0;
	}
	float stepX = lenX/(float)subdX;
	float stepZ = lenZ/(float)subdZ;

	/* build vertex array */
	for( size_t i = 0, offset = 0; i < vertXDim;
		++i,posX += stepX)
	{
		posZ = -lenZ/2.0f;
		for( size_t j = 0; j < vertZDim;
			++j, posZ+= stepZ, ++offset)
		{
			vertexArray[offset].x = posX;
			vertexArray[offset].y = 0.0f;
			vertexArray[offset].z = posZ;
		}
	}
	/* build element array */
	for( size_t i = 0, offset = 0, vIndexTopLeft = 0; i < subdX; ++i, ++vIndexTopLeft)
	{
		for( size_t j = 0; j < subdZ; 
			++j, offset += TRIS_PER_QUAD * VERTS_PER_TRI, ++vIndexTopLeft)
		{
			elementArray[offset]   = vIndexTopLeft; //P1
			elementArray[offset+1] = vIndexTopLeft + 1; //P2
			elementArray[offset+2] = vIndexTopLeft + subdZ + 1; //P3

			elementArray[offset+3] = elementArray[offset+1]; //P2
			elementArray[offset+4] = vIndexTopLeft + subdZ + 2; //P4
			elementArray[offset+5] = elementArray[offset+2]; //P3
		}
	}

	IndexedVertices result;
	result.elementCount = elmCount;
	result.vertexCount = vertCount;
	result.elements = elementArray;
	result.vertices = vertexArray;
	return result;
}
//////////////////////////////////////////////////////////////////////////////
Vertices tesselationBasePlaneSV(size_t basePatchCount, size_t originalResolutionX, size_t originalResolutionZ
	, float& resSpacingX, float& resSpacingZ)
{	
	size_t patchCountZ;
	size_t patchCountX;

	/*
	First, we need to compute the number of patch-divisions in x and z.
	basePatchCount is assigned to the smaller dimension, then a patchCount
	for the bigger dimension is chosen, approximating the original ratio as good as possible.
	As we can only approximate the orig. ratio, the grid-spacing may be a different for x and z.
	*/
	{
		size_t maxDim = std::max(originalResolutionX, originalResolutionZ);
		size_t minDim = std::min(originalResolutionX, originalResolutionZ);

		float ratio = (float)maxDim/(float)minDim;
		size_t maxPatchCount = (size_t)ceil(basePatchCount * ratio);
		float stepMin = 1.0f/basePatchCount;
		float stepMax = ratio/maxPatchCount;
	
		if(originalResolutionX > originalResolutionZ)
		{
			resSpacingX = stepMax;
			resSpacingZ = stepMin;
			patchCountX = maxPatchCount;
			patchCountZ = basePatchCount;
		}
		else
		{
			resSpacingZ = stepMax;
			resSpacingX = stepMin;
			patchCountZ = maxPatchCount;
			patchCountX = basePatchCount;
		}
	}
	/*
	Use computed values to generate the vertice-grid.
	*/
	size_t vertCount = patchCountX * patchCountZ;
	vertex3f *vertexArray = new vertex3f[vertCount]; 
	float posX = 0.0;
	float posZ = 0.0;
	//+ 1 row offset (first vert is lower left corner)
	posX += resSpacingX;

	for( size_t i = 0, offset = 0; i < patchCountX;
		++i, posX += resSpacingX)
	{
		posZ = 0.0;
		for( size_t j = 0; j < patchCountZ;
			++j, posZ += resSpacingZ, ++offset)
		{
			vertexArray[offset].x = posX;
			vertexArray[offset].y = 0.0f;
			vertexArray[offset].z = posZ;
		}
	}

	Vertices result;
	result.vertexCount = vertCount;
	result.vertices = vertexArray;
	return result;
}