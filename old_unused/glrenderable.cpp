#include "glrenderable.h"

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
////////////////////////////////////////////////////////////
/* TerrainRenderable definitions */
TerrainRenderable::TerrainRenderable()
	:sh(nullptr)
	,hasMap(false)
{
	glGenVertexArrays(1, &h_vao);
	glBindVertexArray(h_vao);
}
TerrainRenderable::~TerrainRenderable()
{
	resetRenderbufferResources();

	glBindVertexArray(0);
	glDeleteVertexArrays(1, &h_vao);	
}

void TerrainRenderable::updateViewProjUnif(glm::mat4 mat)
{
	assert(sh && "Renderable has no shader!");
	glUniformMatrix4fv(sh->getUniformLocation("pvm"),
		1,GL_FALSE,glm::value_ptr(mat));
}

void TerrainRenderable::render()
{	
	assert(sh && "Renderable has no shader!");
	glDrawArrays(GL_PATCHES,0,drawCount);
}

void TerrainRenderable::setShader(GLShader* psh)
{
	assert(psh && psh->isUsable() && "Assignment of uncompiled shader!");
	sh = psh;
	glUseProgram(sh->getProgram());
	updateViewProjUnif(glm::mat4(1.0));
}

static Vertices getPatches( const size_2D size)
{
	size_t npX = size.x/MAX_TESS_LEVEL;
	size_t npZ = size.z/MAX_TESS_LEVEL;

	float patchSpacingX = 1.0f/npX;
	float patchSpacingZ = 1.0f/npZ;

	size_t vertCount = npX * npZ;
	vertex3f *vertexArray = new vertex3f[vertCount]; 
	float posX = 0.0;
	float posZ = 0.0;

	for( size_t i = 0, offset = 0; i < npX;
		++i, posX += patchSpacingX)
	{
		posZ = 0.0;
		for( size_t j = 0; j < npZ;
			++j, posZ += patchSpacingZ, ++offset)
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

void TerrainRenderable::initRenderbufferResources(const size_2D size, cudaGraphicsResource_t& imageResource)
{
	resetRenderbufferResources();

	//Generate mesh
	Vertices meshinfo = getPatches(size);

	glBindVertexArray(h_vao);
	glGenBuffers(1,&h_meshVerts);
	// VBO
	glBindBuffer(GL_ARRAY_BUFFER,h_meshVerts);
	glBufferData(GL_ARRAY_BUFFER,meshinfo.vertexCount * sizeof(vertex3f)
		, (float*)(meshinfo.vertices), GL_STATIC_DRAW); 
	// VAO
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,0); 
	glBindBuffer(GL_ARRAY_BUFFER,0);
	//cleanup
	drawCount = meshinfo.vertexCount;
	glBindVertexArray(0);
	Vertices::clear(meshinfo);

	//Generate texture
	glGenTextures(1,&h_dataTexture);
	glBindTexture(GL_TEXTURE_2D, h_dataTexture);

	//Setup texture ; G: material-spec, R: height
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,size.x,size.z,0,GL_RGBA,GL_FLOAT,NULL);
	glBindTexture(GL_TEXTURE_2D,0);

	//Register texture with cuda
	checkCudaErrors(cudaGraphicsGLRegisterImage(
		&imageResource,h_dataTexture,GL_TEXTURE_2D,cudaGraphicsRegisterFlagsSurfaceLoadStore));
	glUseProgram(sh->getProgram());

	//Update shader uniforms
	glUniform1f(sh->getUniformLocation("gridSpacing"),160.0);
	glUniform1i(sh->getUniformLocation("scaleFactor"),1);
	glUniform1f(sh->getUniformLocation("patchSize"),64.0);
	glUniform1i(sh->getUniformLocation("pixelsPerEdge"),3);
	glUniform1f(sh->getUniformLocation("heightStep"),
		0.1f * 2 * 65536);
	glUniform2i(sh->getUniformLocation("viewportDim"),800,600);

	//Sampler
	GLuint sampler;
	glGenSamplers(1,&sampler);
	glSamplerParameteri(sampler,GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
	glSamplerParameteri(sampler,GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
	glSamplerParameteri(sampler,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glSamplerParameteri(sampler,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
#define dataTextureTU 0
	glActiveTexture(GL_TEXTURE0 + dataTextureTU);
	glBindTexture(GL_TEXTURE_2D, h_dataTexture);
	glBindSampler(dataTextureTU,sampler);	
	glUniform1i(sh->getUniformLocation("dataTexture"),dataTextureTU);

	//store some stuff
	dataTextureResource = imageResource;
	hasMap = true;
}

void TerrainRenderable::resetRenderbufferResources()
{
	if(!hasMap)
		return;
	hasMap = false;	

	//Cleanup mesh buffer
	glBindBuffer(GL_ARRAY_BUFFER,0);
	glDeleteBuffers(1, &h_meshVerts);
	//Reset all objects related to the texture		
	checkCudaErrors(cudaGraphicsUnregisterResource(dataTextureResource));
	glBindTexture(GL_TEXTURE_2D,h_dataTexture);
	glDeleteTextures(1,&h_dataTexture);
}
