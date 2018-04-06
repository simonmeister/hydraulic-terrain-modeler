#ifndef TERRAIN_MESH
#define TERRAIN_MESH

#include <string>
#include <vector>

#include "3rdParty\glew\glew.h"
#include <cuda_runtime.h>
#include "util\global.h"
#include "graphics\glshader.h"

class TerrainMesh
{
public:
	TerrainMesh();
	virtual ~TerrainMesh();

	void init(size_2D initSize);

	//Handles are invalid if initialized is false!
	inline bool isInitialized() const
	{ return initialized ;}

	//glDrawArrays and bind/unbind resources before and after
	void render();

	/*For a faster rendering of multiple passes: binding and unbinding
	is managed in prepare/end multipass render. renderPass only calls one function: glDrawArrays*/
	void beginMultipassRender();
	void renderPass();
	void endMultipassRender();
	

	inline GLuint getHeightTextureGLHandle() const 
	{ return h_heightTexture;}

	inline cudaGraphicsResource_t getHeightImageCudaHandle() const
	{ return heightImage;}

	inline GLuint getMaterialTextureGLHandle() const
	{ return h_matTexture;}

	inline cudaGraphicsResource_t getMaterialImageCudaHandle() const
	{ return matImage;}

	inline size_2D getSize() const
	{ return size; }

	void clear();

protected:

	GLuint h_vao;
	GLuint h_heightTexture;
	GLuint h_matTexture;
	GLuint h_meshVerts;

	unsigned int drawCount;
	bool initialized;
	cudaGraphicsResource_t heightImage;
	cudaGraphicsResource_t matImage;
	size_2D size;
};

#endif // TERRAIN_MESH
