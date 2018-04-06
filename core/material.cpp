#include "material.h"
#include "util\global.h"

//------------------------------------------------------------------------
// Material definitions
//------------------------------------------------------------------------

Material::Material(unsigned int materialID, const rgb& c, const rgb& sc, float d)
	: ID(materialID)
	, color(c)
	, specularColor(sc)
	, useCount(0)
	, density(d)
{}

//------------------------------------------------------------------------
// Material Manager definitions
//------------------------------------------------------------------------

MaterialManager::MaterialManager()
	: texturesLoaded(false)
{}

MaterialManager::~MaterialManager()
{
	if(texturesLoaded)
	{
		glDeleteTextures(1,&h_colorBuf);
	}
}

Material* MaterialManager::getMaterial(unsigned int id)
{ 
	assert( materials.size() > id);

	return materials[id];
}

void MaterialManager::addMaterial( const rgb& color, const rgb& specColor, float density)
{
	//create new mat
	Material* mat = new Material(materials.size(),color, specColor, density);
	materials.push_back(mat);
	streamDataToTexture();
}

void MaterialManager::setMaterialProp(unsigned int id, float density)
{
	assert(materials.size() > id);

	materials[id]->density = density;

	streamDataToTexture();
}
void MaterialManager::setMaterialColor(unsigned int id, const rgb& color, const rgb& specColor)
{
	assert(materials.size() > id);

	materials[id]->color = color;
	materials[id]->specularColor = specColor;

	streamDataToTexture();
}
bool MaterialManager::removeMaterial(unsigned int id)
{
	assert(materials.size() > id);

	if(materials[id]->useCount > 0 )
		return false;

	//remove element if unused
	int i = 0;
	for( auto iter = materials.begin(); iter != materials.end(); ++iter, ++i)
	{
		if ( i == id )
		{
			materials.erase(iter);
			break;
		}
	}
	//reassign IDs
	for(unsigned int i = 0; i < materials.size(); ++i)
	{
		materials[i]->setID(i);
	}
	streamDataToTexture();

	return true;
}

void MaterialManager::streamDataToTexture()
{
	if(texturesLoaded)
	{
		glDeleteTextures(1,&h_colorBuf);
	}

	//Convert color data
	float* colorData = new float[4 * materials.size()];
	for(unsigned int i = 0; i < materials.size(); ++i)
	{
		rgb col = materials[i]->getColor();
		colorData[i * 4    ] = col.r;
		colorData[i * 4 + 1] = col.g;
		colorData[i * 4 + 2] = col.b;
		colorData[i * 4 + 3] = 1.0f;
	}

	//Stream color data
	glGenTextures(1,&h_colorBuf);
	glBindTexture(GL_TEXTURE_1D, h_colorBuf);
	
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage1D(GL_TEXTURE_1D,0, GL_RGBA32F, materials.size(), 0 ,
				 GL_RGBA, GL_FLOAT, colorData);

	delete[] colorData;
	glBindTexture(GL_TEXTURE_1D, 0);

	texturesLoaded = true;
}