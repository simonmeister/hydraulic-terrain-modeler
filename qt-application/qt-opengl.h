#ifndef QT_OPENGL_H
#define QT_OPENGL_H

#include <Windows.h>
#include "3rdParty\glew\glew.h"

#include <QtOpenGL\qgl.h>
#include <qevent.h>

#include "util\global.h"
#include "util\glm.h"
#include "util\textureUnits.h"

#include "graphics\camera.h"
#include "graphics\glshader.h"
#include "graphics\misc.h"
#include "graphics\terrainmesh.h"
#include "core\renderer.h"

class QGLRenderer : public QGLWidget
{
	Q_OBJECT
public:
	QGLRenderer(QWidget* parent = NULL);
	~QGLRenderer();

	inline void display()
	{ updateGL(); }

	TerrainMesh* getMeshRenderable()
	{ return mesh;}

	void setShaded(bool on);

	void uploadUniforms();

	inline void setPixelsPerEdge(unsigned int p)
	{
		ppEdge = p;
		uploadUniforms();
	}

	inline void setInteractiveColor(const glm::vec3& inter)
	{
		markerColor = inter;
		uploadUniforms();
	}

	void resetCam(float initZ);

	inline void setBackgroundColor(const glm::vec3& inter)
	{
		glClearColor(inter.r,inter.g,inter.b,1.0);
		bgColor = inter;
	}

	inline glm::vec3 getBackgroundColor() const
	{ return bgColor; }

	inline bool getWaterVisible() const
	{ return renderWater; }

	inline void setWaterVisible(bool t)
	{ renderWater = t; }

	inline bool getLayersVisible() const
	{ return renderLayers; }

	inline void setLayersVisible(bool t)
	{ renderLayers = t; }

	inline void setColorTextureHandle(GLuint h)
	{colorTextureHandle = h; }
	
	inline GLuint getColorTextureHandle() const
	{ return colorTextureHandle;}

	inline glm::vec3 getInteractiveColor() const
	{ return markerColor; }

	inline unsigned int getPixelsPerEdge() const
	{ return ppEdge; }

	inline unsigned int getLastFPS() const
	{ return lastFPS; }

	inline unsigned int getLastPrimitiveCount() const
	{ return primitiveCount; } 

	inline bool getVsyncOn() const
	{ return (this->format().swapInterval() > 0); }

	void setVsyncOn(bool on);

	inline float getMousePosX() const
	{ return currentMousePosX; }

	inline float getMousePosZ() const
	{ return currentMousePosZ; }

	inline void setBrushRadius(float rad)
	{ brushRadius = rad; uploadUniforms();}

	inline void setConstantMarkerRadius(float rad)
	{constantMarkerRadius = rad;uploadUniforms();}

	inline void setTransparency ( bool doIt)
	{useTransparency = doIt; uploadUniforms(); }

	inline void setTransparencyBase(float val)
	{transparencyBase = val;uploadUniforms();}

	inline void setTransparencyMult(float val)
	{transparencyMult = val;uploadUniforms();}

	inline void setTerrainLighting(unsigned int idx)
	{ terrainLighting = idx;uploadUniforms(); }

	inline void setTerrainMult(float val)
	{ terrainMult = val; uploadUniforms();}

	inline void setOcclusionSamples(unsigned int val)
	{ occlusionSamples = val; uploadUniforms();}

	inline void setWaterBase(glm::vec3 val)
	{ waterBase = val; uploadUniforms();}

	inline void setMarkerColor(glm::vec3 val)
	{ markerColor = val; uploadUniforms();}

	inline void setConstantMarkerColor(glm::vec3 val)
	{ constantMarkerColor = val; uploadUniforms();}

	inline void setBGColor(glm::vec3 val)
	{ bgColor = val; uploadUniforms();}

	inline glm::vec3 getBGColor()
	{ return bgColor;}

	inline void setReflections(bool on)
	{ useReflections = on; uploadUniforms();}

	inline void setShowMarker(bool on)
	{ showMarker = on; uploadUniforms();}

	inline void setShowConstantMarker(bool on)
	{ showConstantMarker = on; uploadUniforms();}

	inline glm::vec3 getWaterBase() const
	{ return waterBase; }

	inline glm::vec3 getMarkerColor() const
	{ return markerColor; }

	inline glm::vec3 getConstantMarkerColor() const
	{ return constantMarkerColor; }

	inline void setConstantMarkerPosition(glm::vec2 pos)
	{ constantMarkerPosition = pos; uploadUniforms();}

	inline void setNavigationSpeed(float s)
	{navigationSpeed = s;}

	inline float getNavigationSpeed() const
	{ return navigationSpeed; }
protected:

	void paintGL();
	void initializeGL();
	void resizeGL( int width, int height);

	void keyPressEvent ( QKeyEvent * event );
	void keyReleaseEvent ( QKeyEvent * event );

	void mouseMoveEvent ( QMouseEvent * event );
	void mouseReleaseEvent ( QMouseEvent * event );
	void mousePressEvent ( QMouseEvent * event );

	void leaveEvent ( QEvent * event );

	void wheelEvent ( QWheelEvent * event );

private slots:

	void navigateUpdate();
private:
	bool init;

	//timer to refresh movement
	QTimer* navTimer;

	//movement states
	bool wDown;
	bool sDown;
	bool aDown;
	bool dDown;
	bool vDown;
	bool mouseDown;
	float moveSpeed;
	float rotateSpeed;
	float navigationSpeed;

	//brush settings
	float brushRadius;
	float currentMousePosX;
	float currentMousePosZ;

	//transformation states
	bool	positionReset;
	bool useTransparency;
	char	buttonClicked;
	bool transfUpdate;
	glm::mat4 projMat;
	int lastx;
	int lasty;

	float	zNear;
	float	zFar ;
	float	fov;

	//viewport dimensions
	int vpx;
	int vpz;

	//performance 
	GLuint h_primitiveCounter;
	unsigned int primitiveCount;
	unsigned int lastFPS;


	//misc options
	bool renderLayers;
	bool renderWater;
	glm::vec3 bgColor;
	bool showMarker;
	bool useReflections;
	bool showConstantMarker;
	//uniforms
	unsigned int ppEdge;
	glm::vec3 markerColor;
	glm::vec2 constantMarkerPosition;
	glm::vec3 constantMarkerColor;
	float constantMarkerRadius;

	float transparencyBase;
	float transparencyMult;
	float terrainLighting;
	float terrainMult;
	int occlusionSamples;
	glm::vec3 waterBase;

	//..
	TerrainMesh* mesh;
	Camera* cam;
	GLShader* layerShader;
	GLShader* waterShader;

	//texture handles
	GLuint colorTextureHandle;
	GLuint waterReflect;

signals:
	void brushScroll(int delta);
	void sourceButton();
};

#endif // QT_OPENGL_H