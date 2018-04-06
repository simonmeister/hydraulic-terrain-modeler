#ifndef MATERIAL_H
#define MATERIAL_H

#include <map>
#include <string>
#include <vector>

#include "3rdParty\glew\glew.h"


struct rgb
{
	rgb(float sr, float sg, float sb)
		: r(sr), g(sg) ,b(sb) {}

	float r;
	float g;
	float b;
};

class MaterialManager;

class Material
{
	friend class MaterialManager;
public:
	Material(unsigned int materialID, const rgb& color, const rgb& specularColor, float density);
	virtual ~Material() {}

	inline void incrementUseCount()
	{ ++useCount; }

	inline void decrementUseCount()
	{ --useCount; }

	inline unsigned int getUseCount() const
	{ return useCount;}

	inline rgb getColor() const 
	{ return color; }

	inline rgb getSpecular() const
	{ return specularColor; }

	inline unsigned int getID() const
	{ return ID; }

	inline float getDensity() const
	{ return density; }

protected:
	inline void setID(unsigned int ID)
	{ this->ID = ID; }

private:
	rgb color;
	rgb specularColor;
	float density;

	unsigned int useCount;
	unsigned int ID;
};

class MaterialManager
{
public:
	MaterialManager();
	virtual ~MaterialManager();

	void addMaterial(const rgb& color, const rgb& specColor, float density);

	void setMaterialProp(unsigned int id, float density);
	void setMaterialColor(unsigned int id, const rgb& color, const rgb& specColor);

	//returns true if use count was 0 and material was removed
	bool removeMaterial(unsigned int id);

	inline unsigned int getMaterialCount()
	{ return materials.size(); }

	Material* getMaterial(unsigned int id);

	inline GLuint getColorTextureHandle()
	{ return h_colorBuf; }

protected:
	void streamDataToTexture();

private:

	//std::map < std::string, Material* > matsByName;
	std::vector < Material*> materials;

	GLuint h_colorBuf;
	bool texturesLoaded;
};


#endif MATERIAL_H