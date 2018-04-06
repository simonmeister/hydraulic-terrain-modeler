#include "terrainmesh.h"
#include "util\global.h"

#include <iostream>
#include <cmath>
#include <cuda_gl_interop.h>
#include "3rdParty\cuda-helper\helper_cuda.h"
#include "util/textureUnits.h"



#define MAX_TESS_LEVEL 64

TerrainMesh::TerrainMesh()
	: initialized( false ) 
{}

TerrainMesh::~TerrainMesh() 
{
	clear();
}

void TerrainMesh::clear()
{
	if( !initialized) 
		return;


	//clear mesh buffers
	glBindBuffer(GL_ARRAY_BUFFER,0);
	glDeleteBuffers(1, &h_meshVerts);
	glDeleteVertexArrays(1,&h_vao);

	//Reset all objects related to the texture		
	checkCudaErrors(cudaGraphicsUnregisterResource(heightImage));
	glDeleteTextures(1,&h_heightTexture);

	checkCudaErrors(cudaGraphicsUnregisterResource(matImage));
	glDeleteTextures(1,&h_matTexture);

	initialized = false;
}

void TerrainMesh::render() 
{
	assert(initialized && "Rendercall to uninitialized Terrainmesh!");

	beginMultipassRender();
	renderPass();
	endMultipassRender();
}

void TerrainMesh::beginMultipassRender()
{
	assert(initialized && "Rendercall to uninitialized Terrainmesh!");

	glPatchParameteri( GL_PATCH_VERTICES, 1 );

	// bind textures
	glActiveTexture(GL_TEXTURE0 + TU_TERRAINMESH_HEIGHT);
	glBindTexture(GL_TEXTURE_2D, h_heightTexture);

	glActiveTexture(GL_TEXTURE0 + TU_TERRAINMESH_MATERIAL);
	glBindTexture(GL_TEXTURE_2D, h_matTexture);
	// bind VAO
	glBindVertexArray(h_vao);
}

void TerrainMesh::renderPass()
{
	assert(initialized && "Rendercall to uninitialized Terrainmesh!");

	glDrawArrays(GL_PATCHES,0,drawCount);
}

void TerrainMesh::endMultipassRender()
{
	assert(initialized && "Rendercall to uninitialized Terrainmesh!");
	glActiveTexture(GL_TEXTURE0 + TU_TERRAINMESH_HEIGHT);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0 + TU_TERRAINMESH_MATERIAL);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);
}

void TerrainMesh::init(size_2D s) 
{
	clear();
	size = s;

	/* Generate base tesselation mesh */
	// round patches up 
	int nopX = (int) ceil(size.x /(float) MAX_TESS_LEVEL);
	int nopZ = (int) ceil(size.z /(float) MAX_TESS_LEVEL);
	int numPatches = nopX * nopZ;

	drawCount = numPatches;
	float *vertices = new float[2 * drawCount];
	unsigned int  *indices = new unsigned int[numPatches];

	int patchNumber;
	for (int i = 0; i < nopX; ++i) 
	{
	
		for (int j = 0; j < nopZ; ++j) 
		{	
			patchNumber = i * nopZ + j;

			vertices[patchNumber * 2    ] = (i * MAX_TESS_LEVEL ) * 1.0f / size.x ;
			vertices[patchNumber * 2 + 1] = (j * MAX_TESS_LEVEL) * 1.0f/ size.z ;				
		}
	}
	//Generate vao
	glGenVertexArrays(1, &(h_vao));
	glBindVertexArray(h_vao);
	//Generate vbo and stream vbo data
	glGenBuffers(1, &h_meshVerts);
	glBindBuffer(GL_ARRAY_BUFFER, h_meshVerts);
	glBufferData(GL_ARRAY_BUFFER,sizeof(float) * drawCount * 2,
					vertices,GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, 0, 0, 0);
	//Free memory
	delete[] vertices;
	delete[] indices;

	/* Setup textures with appropriate size*/
	//height texture
	glGenTextures(1, &h_heightTexture);
	glBindTexture(GL_TEXTURE_2D, h_heightTexture); 
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, size.height, size.width, 
					0, GL_RED, GL_FLOAT,NULL); 
	glBindTexture(GL_TEXTURE_2D, 0);
	//material-id texture
	glGenTextures(1, &h_matTexture);
	glBindTexture(GL_TEXTURE_2D, h_matTexture); 
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, size.height, size.width, 
					0, GL_RED, GL_UNSIGNED_BYTE,NULL); 

	glBindTexture(GL_TEXTURE_2D, 0);
	//Register textures with cuda
	checkCudaErrors(cudaGraphicsGLRegisterImage(
		&heightImage,h_heightTexture,GL_TEXTURE_2D,cudaGraphicsRegisterFlagsSurfaceLoadStore));
	checkCudaErrors(cudaGraphicsGLRegisterImage(
		&matImage,h_matTexture,GL_TEXTURE_2D,cudaGraphicsRegisterFlagsSurfaceLoadStore));

	initialized = true;
}

