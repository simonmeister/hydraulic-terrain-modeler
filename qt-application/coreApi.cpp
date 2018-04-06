#include "coreApi.h"

QtHyTM::QtHyTM(QGLRenderer* render)
	: renderer(render)
	, currentTicks(30)
	, simulationActive(false)
	, terrainLoaded(false)
	, forceUpdate(false)
	,cWSim(0)
	,cSSim(0)
	,cSimMan(0)
	, cTerrain(0)
{
	timer = new QTimer();

	connect(timer,SIGNAL(timeout()), this, SLOT(loop()));
	timer->start(0);
	cMatMan = new MaterialManager();
	rbrush = new RadialBrush();
}

QtHyTM::~QtHyTM()
{
	if(cSimMan)
	{
		delete cSimMan;
		delete cTerrain;
		delete cWSim;
		delete cSSim;
	}
}

void QtHyTM::prepareResolution(size_2D rez)
{
	//if that is not the first time...
	if(cSimMan)
	{
		delete cSimMan;
		WaterPipeSimulator *pTemp = cWSim;
		SourceSimulator* sTemp = cSSim;

		cWSim = new WaterPipeSimulator();
		cWSim->copySettings(*pTemp);

		cSSim = new SourceSimulator();
		cSSim->copySettings(*sTemp);


		delete pTemp;
		delete sTemp;

		delete cTerrain;
	}
	else
	{
		cWSim = new WaterPipeSimulator();
		cSSim = new SourceSimulator();
	}
	//prepare
	terrainLoaded = true;
	cTerrain = new Terrain(rez);

	TerrainMesh* m = renderer->getMeshRenderable();
	m->init(rez);
	cTerrain->connectDataReceivers(m->getHeightImageCudaHandle(),
		m->getMaterialImageCudaHandle());

	std::vector<Simulator*> sims;

	sims.push_back(cSSim);
	sims.push_back(cWSim);

	SharedCudaResources* rcman = new SharedCudaResources();
	cSimMan = new SimulationManager(cTerrain->getWorkHandle(),rcman,sims);
	rbrush->setTerrain(cTerrain->getWorkHandle());
}

void QtHyTM::paint(float x, float z)
{
	if(terrainLoaded)
	{
		rbrush->paint(x,z);
		forceUpdate = true;
	}
}

void QtHyTM::addEmptyLayer(size_2D s,glm::vec3 c, float matDensity)
{
	if(!cTerrain || !cTerrain->isInitialized())
	{
		prepareResolution(s);
	}

	cMatMan->addMaterial(rgb(c.r,c.g,c.b),rgb(c.r,c.g,c.b),matDensity);
	cTerrain->addEmptyLayer(cMatMan->getMaterial(cTerrain->getLayerCount()));

	renderer->setColorTextureHandle(cMatMan->getColorTextureHandle());
}

void QtHyTM::removeLayer(unsigned int id)
{
	cTerrain->removeLayer(id);
	cMatMan->removeMaterial(id);

	if(!cTerrain->isInitialized())
	{
		terrainLoaded = false;
		cTerrain->disconnectDataReceivers();
		renderer->getMeshRenderable()->clear();
	}
}

void QtHyTM::addLayer(size_2D s, float* src, float scale, glm::vec3 c, float matDensity)
{
	if(!cTerrain || !cTerrain->isInitialized())
	{
		prepareResolution(s);
	}

	cMatMan->addMaterial(rgb(c.r,c.g,c.b),rgb(c.r,c.g,c.b),matDensity);

	cTerrain->addLayer(src,cTerrain->getSize(),scale,cMatMan->getMaterial(cTerrain->getLayerCount()));
	renderer->setColorTextureHandle(cMatMan->getColorTextureHandle());
	forceUpdate = true;
}


void QtHyTM::setMaterialProp(unsigned int id, float d)
{
	cMatMan->setMaterialProp(id,d);
}

Material* QtHyTM::getMaterial(unsigned int id)
{
	return cMatMan->getMaterial(id);
}
void QtHyTM::setMaterialColor(unsigned int id, glm::vec3 c)
{
	cMatMan->setMaterialColor(id, rgb(c.r,c.g,c.b),rgb(c.r,c.g,c.b));
}

void QtHyTM::loop()
{
	static unsigned long nextTick = GetTickCount();
	int loops = 0;
	static int frames = 0;

	while( GetTickCount() > nextTick && loops < MAX_FRAMESKIP) 
	{
		//Compute FPS
		static int frames = 0;
		++frames;
		static int lastTime = timeGetTime();
		int now = timeGetTime();
		if( now - lastTime > 1000)
		{
			lastTime = now;
			actualTicks = frames;

			frames = 0;
		}

		if(terrainLoaded && cTerrain->isInitialized())
		{
			if(forceUpdate)
				cTerrain->forceUpdateDataReceivers();
			else
				cTerrain->updateDataReceivers();
		}

		if(terrainLoaded && simulationActive)
			cSimMan->update();

		nextTick += 1000/currentTicks;
		loops++;
	}

	renderer->updateGL();
}

float* QtHyTM::write(size_2D& size, bool writeWater)
{
	return cTerrain->write(size,writeWater);
}

float* QtHyTM::writeLayer(size_2D& size, unsigned int layerID)
{
	return cTerrain->writeLayer(size,layerID);
}
float* QtHyTM::writeWater(size_2D& size)
{
	return cTerrain->writeWater(size);
}

void QtHyTM::addLayerHeight(const float* source,const size_2D size, unsigned int layerID, float heightScale)
{
	cTerrain->addLayerHeight(source,size,layerID,heightScale);
}

void QtHyTM::addWater(const float* source,const size_2D size, float heightScale)
{
	cTerrain->addWater(source,size,heightScale);
}