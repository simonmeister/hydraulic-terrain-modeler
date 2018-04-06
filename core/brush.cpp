#include "brush.h"


extern void cw_radialSourcing(floatMem& heights, size_2D gridSize,
	                        float perCellAmount, 
							size_2D fromIdx, size_2D toIdx, size_2D posIdx,
							float radius, float hardness, bool useFalloff);


Brush::Brush()
	: ter(0)
	, intensity(DEFAULT_INTENSITY)
{}

//------------------------------------------------------------------------
// Radial Brush
//------------------------------------------------------------------------
RadialBrush::RadialBrush()
	: radius(DEFAULT_RADIUS)
	, brushLayer( -1)
	, hardness(DEFAULT_HARDNESS)
{}

void RadialBrush::paint(float px, float pz)
{
	assert(px <= 1.0f && px >= 0.0f && "Out of range");
	assert(pz <= 1.0f && pz >= 0.0f && "Out of range");

	//select layer to paint at
	floatMem receiver;
	if(brushLayer == -1)
		receiver = ter->water();
	else
	{
		receiver = ter->layer(brushLayer);
	}


	//calculate execution domain in normalized source coordinates
	float nxfrom,nzfrom,nxto, nzto;

	float rx,rz;
	//scale down the along larger dimension, as it will be scaled up to strong by the
	//second level of calculations
	if( ter->size().x >  ter->size().z)
	{
		float aspectRatio = ter->size().x / (float)ter->size().z;
		rz = radius;
		rx = radius/aspectRatio;
	}
	else
	{
		float aspectRatio = ter->size().z / (float)ter->size().x;
		rx = radius;
		rz = radius/aspectRatio;
	}
	
	nxfrom = std::max<float>(0.0, px - rx );
	nzfrom = std::max<float>(0.0, pz - rz);
	nxto =	 std::min<float>(1.0, px + rx);
	nzto =	 std::min<float>(1.0, pz + rz );

	//normalized source coordinates to grid coordinates
	size_t xfrom,xto,zfrom,zto,xpos, zpos;

	xfrom =	(size_t)	(nxfrom * (ter->size().x-1));
	xto =	(size_t)	(nxto * (ter->size().x-1));
	zfrom =	(size_t)	(nzfrom * (ter->size().z-1));
	zto =	(size_t)	(nzto * (ter->size().z-1));

	xpos =	(size_t)	(px * (ter->size().x-1));
	zpos =	(size_t)	(pz  * (ter->size().z-1));

	cw_radialSourcing(receiver, ter->size(),intensity,
				size_2D(xfrom,zfrom),size_2D(xto,zto), size_2D(xpos,zpos), radius, hardness, true);
}

/*
//------------------------------------------------------------------------
// Rectangular Brush 
//------------------------------------------------------------------------

RectBrush::RectBrush()
	: lenx(DEFAULT_SIDELEN)
	, lenz(DEFAULT_SIDELEN)
{}

void RectBrush::paint(size_2D p)
{
	//calculate execution domain in normalized source coordinates
	float nxfrom,nzfrom,nxto, nzto;
	nxfrom = std::max<float>(0.0, p.x - lenx/2.0f);
	nzfrom = std::max<float>(0.0, p.z - lenz/2.0f);
	nxto =	 std::min<float>(1.0, p.x + lenx/2.0f);
	nzto =	 std::min<float>(1.0, p.z + lenz/2.0f);

	//normalized source coordinates to grid coordinates
	size_t xfrom,xto,zfrom,zto;

	xfrom =	(size_t)	(nxfrom * (ter->size().x-1));
	xto =	(size_t)	(nxto * (ter->size().x-1));
	zfrom =	(size_t)	(nzfrom * (ter->size().z-1));
	zto =	(size_t)	(nzto * (ter->size().z-1));

	
	cw_rectSourcing(ter->water(), ter->size(), intensity,
				size_2D(xfrom,zfrom),size_2D(xto,zto));
}

*/