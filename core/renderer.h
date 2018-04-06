#ifndef RENDERER_H
#define RENDERER_H

class Renderer
{
public:
	Renderer(){}
	virtual ~Renderer(){}

	virtual void display() = 0;
	virtual TerrainMesh* getMeshRenderable() = 0;
};

#endif //RENDERER_H