#ifndef MESHGEN_H
#define MESHGEN_H

///////////////////////////////////////////////////////////////////////
//OpenGL groundmesh generation

struct vertex3f
{
	float x;
	float y;
	float z;
};

class IndexedVertices
{
public:
	IndexedVertices();
	virtual ~IndexedVertices(){}
	static void clear(IndexedVertices& t);

	size_t vertexCount;
	vertex3f* vertices;

	size_t elementCount;
	unsigned int* elements;
};

class Vertices
{
public:
	Vertices();
	virtual ~Vertices(){}
	static void clear(Vertices& t);
	size_t vertexCount;

	vertex3f* vertices;
};

//NOTE: Parameter symmetric specifies wether plane is centered at the origin or 
//goes from 0,0 to lenX,lenZ

//Returns a plane made up of quads. mostly useful as tesselator patches
//should be drawn using glDrawElements(GL_PATCHES...)
IndexedVertices quadPlane(size_t subdX, size_t subdZ,
	float lenX = 1.0f, float lenZ = 1.0f, bool symmetric = true);

//Returns a plane made up of quads that are split into triangles
//should be drawn using glDrawElements(GL_TRIANGLES...)
IndexedVertices trianglePlane(size_t subdX, size_t subdZ,
	float lenX = 1.0f, float lenZ = 1.0f, bool symmetric = true);

//Returns a plane where every gridcell is defined by the vertex at its lower left corner
//should be drawn using glDrawArrays(GL_TRIANGLES...)
Vertices tesselationBasePlaneSV(size_t basePatchCount, size_t originalResolutionX, size_t originalResolutionZ
	, float& resSpacingX, float& resSpacingY);


//#define MAX_TESS 64
//Vertices getPlanePatch();


#endif //MESHGEN_H