#ifndef BRUSH_H
#define BRUSH_H

#include "terrain.h"
#include "util/global.h"
#include "sourcing_types.h"

const float DEFAULT_INTENSITY = 1.0;

class Brush
{
public:
	Brush();
	virtual ~Brush(){}

	inline void setTerrain(TerrainWorkHandle* terrain)
	{ter = terrain;}

	virtual void paint(float px, float py){}

	inline void setIntensity(float val)
	{ intensity = val;}

	inline float getIntensity()
	{ return intensity; }

protected:
	float intensity;
	TerrainWorkHandle* ter;
};



const float DEFAULT_RADIUS = 10;
const float DEFAULT_HARDNESS = 0.3f;

class RadialBrush : public Brush
{
public:
	RadialBrush();
	~RadialBrush(){}

	virtual void paint(float px, float py);

	inline void setRadius(float val)
	{ radius = val; }

	inline void setHardness(float fo)
	{ 
		assert(fo > 0.0f && fo <= 1.0f && "Out of range");
		hardness = fo; 
	}

	inline float getHardness()
	{ return hardness;}

	inline float getRadius()
	{ return radius; }

	inline void setBrushLayer(size_t layer)
	{ brushLayer = layer;}

	inline void setBrushWater()
	{ brushLayer = -1; }

	
	inline int getBrushLayer()
	{ return brushLayer; }

private:
	float radius;
	float hardness;
	//-1 is water
	int brushLayer;

};
/*
const float DEFAULT_SIDELEN = 10;

class RectBrush : public Brush
{
public:
	RectBrush();
	~RectBrush(){}

	virtual void paint(size_2D position);

	inline void setSideLengths(float lenx, float lenz)
	{ this->lenx = lenx; this->lenz = lenz;}

	inline void getSideLengths(float& slenx, float& slenz)
	{ slenx = lenx; slenz = lenz; }

private:
	float lenx;
	float lenz;
};*/
#endif //BRUSH_H