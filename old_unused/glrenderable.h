#ifndef RENDERABLE_H
#define RENDERABLE_H

//HyTM includes
#include "glshader.h"
#include "core/terrain.h"

#include "util/global.h"

#include "util/glm.h"
#include "3rdParty/glew/glew.h"
#include <cuda_runtime.h>
#include "3rdParty\cuda-helper\helper_cuda.h"
#include "cuda_gl_interop.h"


#define MAX_TESS_LEVEL 64
struct vertex3f
{
	float x;
	float y;
	float z;
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



class TerrainRenderable
{
public:
	TerrainRenderable();
	virtual ~TerrainRenderable();

	virtual void updateViewProjUnif(glm::mat4 mat);
	virtual void render();

	virtual void setShader(GLShader* sh);
	inline virtual GLShader* getShader() const { return sh; }

	inline virtual bool isVisible() const {return hasMap;}
	inline virtual void setVisible(bool vis) {}

	//Write a cude handle to a new gl buffer into "imageResource". This buffer is now
	//passed to the shaders inside of render function.
	void initRenderbufferResources(const size_2D, cudaGraphicsResource_t& imageResource);
	void resetRenderbufferResources();

private:
	GLShader* sh;
	bool hasMap;

	size_t drawCount;
	GLuint h_vao;
	GLuint h_meshVerts;
	GLint h_uniformViewProj;
	GLuint h_dataTexture;
	cudaGraphicsResource_t dataTextureResource;
	GLuint sampler;
};

#endif //RENDERABLE_H
