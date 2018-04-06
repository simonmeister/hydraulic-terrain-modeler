#ifndef CORE_API_H
#define CORE_API_H

#include <qtimer.h>
#include <qobject.h>
#include "qt-opengl.h"

#include "core\brush.h"
#include "core\material.h"
#include "core\simulationManager.h"
#include "core\sourceSimulator.h"
#include "core\terrain.h"
#include "core\waterPipeSimulator.h"
#include "core\simulator.h"

const int MAX_FRAMESKIP = 10;

class QtHyTM : public QObject
{ 
	Q_OBJECT
public:
	QtHyTM(QGLRenderer* render);
	~QtHyTM();

	inline void setSimulationActive(bool on)
	{ simulationActive = on; }

	inline bool getSimulationActive()
	{ return simulationActive; }

	inline unsigned int getActualTicks() const
	{ return actualTicks;}

	inline unsigned int getCurrentTicks() const
	{ return currentTicks; }

	inline void setCurrentTicks(unsigned int ticks)
	{ currentTicks = ticks; }

	inline RadialBrush* getBrush()
	{ return rbrush; }

	inline SourceSimulator* getSSim()
	{ return cSSim; }

	inline WaterPipeSimulator* getWSim()
	{ return cWSim; }

	Material* getMaterial(unsigned int id);

	inline bool getTerrainLoaded() const
	{ return terrainLoaded; }

	void paint(float x, float z);

	//undefined if getTerrainLoaded returns false
	inline size_2D getRez() const
	{ return cTerrain->getSize(); }

	// terrain layer handling ----------------------
	//material count always equals layer count.
	
	void addEmptyLayer(size_2D s, glm::vec3 matColor, float matDensity);
	void removeLayer(unsigned int id);
	void addLayer(size_2D s,float* src, float scale , glm::vec3 matColor, float matDensity);
	void setMaterialProp(unsigned int id,float density);
	void setMaterialColor(unsigned int id, glm::vec3 color);

	float* write(size_2D& size, bool writeWater = false);
	float* writeLayer(size_2D& size, unsigned int layerID);
	float* writeWater(size_2D& size);

	void addLayerHeight(const float* source,const size_2D size, unsigned int layerID, float heightScale);
	void addWater(const float* source,const size_2D size, float heightScale);

	inline void resetWater()
	{ cTerrain->resetWater(); }

private slots:
	void loop();

private:	
	void prepareResolution(size_2D rez);
	bool simulationActive;
	bool forceUpdate;
	bool terrainLoaded;

	QTimer* timer;

	unsigned int currentTicks;
	unsigned int actualTicks;

	Terrain* cTerrain;
	WaterPipeSimulator* cWSim;
	SourceSimulator* cSSim;
	SimulationManager* cSimMan;

	RadialBrush* rbrush;

	MaterialManager* cMatMan;

	QGLRenderer* renderer;
};

#endif //CORE_API_H