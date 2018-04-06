#include "qt-opengl.h"
#include "core/imageIO.h"
#include <iostream>
#include "util.h"

#include <qtimer.h>

QGLRenderer::QGLRenderer(QWidget* parent)
	:  QGLWidget(parent)
	, zNear(0.1f)
	, zFar(30000.0f)
	, fov(45.0f)
	, moveSpeed(30.0f)
	, rotateSpeed(0.1f)
	, lastFPS(0)
	, primitiveCount(0)
	, transfUpdate(true)
	, ppEdge(3)
	, colorTextureHandle(0)
	, renderLayers(true)
	, renderWater(true)
	, wDown(false)
	, aDown(false)
	, sDown(false)
	, dDown(false)
	, mouseDown(false)
	, currentMousePosX(0.0f)
	, currentMousePosZ(0.0f)
	, brushRadius(0.1f)
	, constantMarkerRadius(0.1f)
	, init (false)
	, vDown(false)
	, transparencyBase(10.0)
	, transparencyMult(1.0)
	, terrainLighting(0)
	, terrainMult(0.5f)
	, occlusionSamples(8)
	, showMarker(true)
	, showConstantMarker(false)
	, useReflections(true)
	, navigationSpeed(1.0)
	, useTransparency(true)
{
	mesh = new TerrainMesh();
	vpx = this->size().width();
	vpz = this->size().height();

	bgColor = glm::vec3(0.9,0.9,0.9);
	markerColor = glm::vec3(0.2, 0.8 , 1.0);
	constantMarkerColor = glm::vec3(0.8, 0.2 , 1.0);
	constantMarkerPosition = glm::vec2(0.0,0.0);

	waterBase = glm::vec3(0.6,0.6,1.0);

	QGLFormat frmt(QGL::DepthBuffer | QGL::DoubleBuffer | QGL::SampleBuffers);
	frmt.setSwapInterval(0);
	this->setFormat(frmt);

	setFocusPolicy(Qt::ClickFocus);
	setMouseTracking(true);
	navTimer = new QTimer();
	connect(navTimer,SIGNAL(timeout()), this, SLOT(navigateUpdate()));
	navTimer->start(10);
}

void QGLRenderer::setVsyncOn(bool on)
{
//	QGLFormat frmt(QGL::DepthBuffer | QGL::DoubleBuffer | QGL::SampleBuffers);
//	frmt.setSwapInterval(on);
//	this->setFormat(frmt);
}
QGLRenderer::~QGLRenderer()
{
	delete mesh;
}

void QGLRenderer::setShaded(bool on)
{
	if(on)
	{
		glPolygonMode(GL_FRONT,GL_FILL);
	}
	else
	{
		glPolygonMode(GL_FRONT,GL_LINE);
	}
}


void QGLRenderer::paintGL()
{
	//Compute FPS
	static int frames = 0;
	++frames;
	static int lastTime = timeGetTime();
	int now = timeGetTime();
	if( now - lastTime > 1000)
	{
		lastTime = now;
		lastFPS = frames;

		frames = 0;
	}

	//Update pvm if needed
	if(transfUpdate)
	{
		glUseProgram(layerShader->getProgram());
		glUniformMatrix4fv(layerShader->getUniformLocation("pv"),
		1,GL_FALSE,glm::value_ptr(projMat * cam->getCameraTransform()));

		glUseProgram(waterShader->getProgram());
		glUniformMatrix4fv(waterShader->getUniformLocation("pv"),
		1,GL_FALSE,glm::value_ptr(projMat * cam->getCameraTransform()));

		glm::vec3 campos = cam->getPosition();
		glUseProgram(waterShader->getProgram());
		glUniform3f(waterShader->getUniformLocation("cameraPosition"),campos.x, campos.y, campos.z);

		transfUpdate = false;
	}


	

	//Draw 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(!mesh->isInitialized())
		return;

	glBeginQuery(GL_PRIMITIVES_GENERATED, h_primitiveCounter);	

	//bind matinfo
	glActiveTexture(GL_TEXTURE0 + TU_MATERIAL_COLOR);
	glBindTexture(GL_TEXTURE_1D, colorTextureHandle);

	glActiveTexture(GL_TEXTURE0 + TU_WATER_REFLECTIVE);
	glBindTexture(GL_TEXTURE_CUBE_MAP, waterReflect);
	// material pass
	mesh->beginMultipassRender();

	if(renderLayers)
	{
		glUseProgram(layerShader->getProgram());
		mesh->renderPass();
	}

	if(renderWater)
	{
		//water pass
		if(useTransparency)
		{
			glDepthMask(GL_FALSE);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_BLEND); 
		}
		glUseProgram(waterShader->getProgram());
		mesh->renderPass();

		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
	}
	mesh->endMultipassRender();
	//unbind
	glBindTexture(GL_TEXTURE_1D, 0);

	glEndQuery(GL_PRIMITIVES_GENERATED);
	glGetQueryObjectuiv(h_primitiveCounter, GL_QUERY_RESULT, &primitiveCount);
}

void QGLRenderer::initializeGL()
{
	init = true;

	GLenum error = glewInit();
	if( GLEW_OK != error)
	{
		sendError((const char*)glewGetErrorString(error),true);
	}
	if( ! GLEW_VERSION_4_0 )
	{
		sendError("OpenGL 4.0 not supported. Please check drivers and hardware.",true);
	}

	//------------------------------------------------------------------------
	// OpenGL settings
	//------------------------------------------------------------------------

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);
	glClearDepth(1.0);
	glClearColor(bgColor.r,bgColor.g,bgColor.b,1.0);
	glGenQueries(1,&h_primitiveCounter);
	this->format().setSwapInterval(0);
	this->setAutoBufferSwap(true);

	//------------------------------------------------------------------------
	// Initialize camera and load Shaders
	//------------------------------------------------------------------------
	cam = new Camera(glm::vec3(0.0,1700.0,0.0), glm::vec3(0.27,-1.0,0.27));
	projMat = glm::mat4(1.0);

	layerShader = new GLShader();
	layerShader->addShaderFromFile("shader/terrain.tc.fp",GL_TESS_CONTROL_SHADER);
	layerShader->addShaderFromFile("shader/terrain.te.fp",GL_TESS_EVALUATION_SHADER);
	layerShader->addShaderFromFile("shader/terrain.v.fp",GL_VERTEX_SHADER);
	layerShader->addShaderFromFile("shader/terrain.f.fp",GL_FRAGMENT_SHADER);
	layerShader->link();

	waterShader = new GLShader();
	waterShader->addShaderFromFile("shader/water.tc.fp",GL_TESS_CONTROL_SHADER);
	waterShader->addShaderFromFile("shader/water.te.fp",GL_TESS_EVALUATION_SHADER);
	waterShader->addShaderFromFile("shader/water.v.fp",GL_VERTEX_SHADER);
	waterShader->addShaderFromFile("shader/water.f.fp",GL_FRAGMENT_SHADER);
	waterShader->link();	

	if(!(layerShader->isUsable() && waterShader->isUsable()))
	{
		sendError("Failed to load Shaders. Files not found at ./shader or compilation error.",true);
	}

	mesh = new TerrainMesh();

	//------------------------------------------------------------------------
	// Load and stream cubemap for water reflections
	//------------------------------------------------------------------------

	glEnable(GL_TEXTURE_CUBE_MAP);

	glGenTextures(1,&waterReflect);
	glBindTexture(GL_TEXTURE_CUBE_MAP,waterReflect);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT);

	size_2D textureSize;
	float* text[6];
	bool success;
	success = readImageRGB("media/textures/ref_posx.bmp",text[0],textureSize);
	success = readImageRGB("media/textures/ref_negx.bmp",text[1] ,textureSize);
	success = readImageRGB("media/textures/ref_posy.bmp",text[2],textureSize);
	success = readImageRGB("media/textures/ref_negy.bmp",text[3],textureSize);
	success = readImageRGB("media/textures/ref_posz.bmp",text[4],textureSize);
	success = readImageRGB("media/textures/ref_negz.bmp",text[5],textureSize);

	if(!success)
	{
		sendError("Failed to load Textures. Files not found at ./media/textures.",true);
	}

	for(int i = 0; i < 6; ++i)
	{
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, 0, GL_RGB32F, textureSize.height, textureSize.width, 
					0, GL_RGB, GL_FLOAT,text[i]); 
	}
	glBindTexture(GL_TEXTURE_CUBE_MAP,0);


	//------------------------------------------------------------------------
	// Set shader uniforms
	//------------------------------------------------------------------------
	
	glUseProgram(layerShader->getProgram());
	glUniformMatrix4fv(layerShader->getUniformLocation("pv"),
	1,GL_FALSE,glm::value_ptr(projMat * cam->getCameraTransform()));

	glUseProgram(waterShader->getProgram());
	glUniformMatrix4fv(waterShader->getUniformLocation("pv"),
	1,GL_FALSE,glm::value_ptr(projMat * cam->getCameraTransform()));

	glm::vec3 campos = cam->getPosition();
	glUseProgram(waterShader->getProgram());
	glUniform3f(waterShader->getUniformLocation("cameraPosition"),campos.x, campos.y, campos.z);

	//layers
	glUseProgram(layerShader->getProgram());
	glUniform1f(layerShader->getUniformLocation("gridSpacing"),1.0);
	glUniform1f(layerShader->getUniformLocation("patchSize"),64.0);
	glUniform1i(layerShader->getUniformLocation("pixelsPerEdge"),ppEdge);

	glUniform2i(layerShader->getUniformLocation("viewportDim"),800,600);
	glUniform1i(layerShader->getUniformLocation("heightTexture"),TU_TERRAINMESH_HEIGHT);
	glUniform1i(layerShader->getUniformLocation("matTexture"),TU_TERRAINMESH_MATERIAL);
	glUniform1i(layerShader->getUniformLocation("materialColorTexture"),TU_MATERIAL_COLOR);

	//water
	glUseProgram(waterShader->getProgram());
	glUniform1f(waterShader->getUniformLocation("gridSpacing"),1.0);
	glUniform1f(waterShader->getUniformLocation("patchSize"),64.0);
	glUniform1i(waterShader->getUniformLocation("heightTexture"),TU_TERRAINMESH_HEIGHT);
	glUniform1i(waterShader->getUniformLocation("reflectionCubemap"),TU_WATER_REFLECTIVE);

	uploadUniforms();
}


void QGLRenderer::uploadUniforms()
{
	if(!init)
		return;

	int sm = showMarker;
	int csm = showConstantMarker;
	int refl = useReflections;
	int transp = useTransparency;

	glClearColor(bgColor.r,bgColor.g,bgColor.b,1.0);
	//layers
	glUseProgram(layerShader->getProgram());
	glUniform1i(layerShader->getUniformLocation("pixelsPerEdge"),ppEdge);
	glUniform2i(layerShader->getUniformLocation("viewportDim"),vpx,vpz);
	glUniform3f(layerShader->getUniformLocation("markerColor"),markerColor.r,markerColor.g,markerColor.b);
	glUniform1f(layerShader->getUniformLocation("brushRadius"),brushRadius);
	glUniform1i(layerShader->getUniformLocation("terrainLighting"),terrainLighting);
	glUniform1f(layerShader->getUniformLocation("terrainMult"),terrainMult);
	glUniform1i(layerShader->getUniformLocation("occlusionSamples"),occlusionSamples);
	glUniform1i(layerShader->getUniformLocation("showMarker"),sm);
	glUniform1i(layerShader->getUniformLocation("showConstantMarker"),csm);
	glUniform3f(layerShader->getUniformLocation("constantMarkerColor"),constantMarkerColor.r,constantMarkerColor.g,constantMarkerColor.b);
	glUniform2f(layerShader->getUniformLocation("constantMarkerPosition"),constantMarkerPosition.x, constantMarkerPosition.y);
	glUniform1f(layerShader->getUniformLocation("constantMarkerRadius"),constantMarkerRadius);
	
	//water
	glUseProgram(waterShader->getProgram());
	glUniform1i(waterShader->getUniformLocation("pixelsPerEdge"),ppEdge);
	glUniform2i(waterShader->getUniformLocation("viewportDim"),vpx,vpz);
	glUniform3f(waterShader->getUniformLocation("markerColor"),markerColor.r,markerColor.g,markerColor.b);
	glUniform1f(waterShader->getUniformLocation("brushRadius"),brushRadius);
	glUniform1f(waterShader->getUniformLocation("transparencyBase"),transparencyBase);
	glUniform1f(waterShader->getUniformLocation("transparencyMult"),transparencyMult);
	glUniform3f(waterShader->getUniformLocation("waterBase"),waterBase.r,waterBase.g,waterBase.b);
	glUniform1i(waterShader->getUniformLocation("showMarker"),sm);
	glUniform1i(waterShader->getUniformLocation("useReflections"),refl);
	glUniform1i(waterShader->getUniformLocation("useTransparency"),transp);
}

void QGLRenderer::resizeGL( int w, int h)
{
	//recalc fov
	glViewport(0,0,(GLsizei)w,(GLsizei)h);

	vpx = w;
	vpz = h;

	projMat = glm::perspective(45.0f,((float)w/float(h)),zNear,zFar);
	transfUpdate = true;
	uploadUniforms();

	wDown = false;
	sDown = false;
	aDown = false;
	dDown = false;
	vDown = false;
}

void QGLRenderer::keyPressEvent ( QKeyEvent * event )
{
	if(event->isAutoRepeat())
		return;

	if(event->key() == Qt::Key::Key_W)
	{
		wDown = true;
	}
	else if(event->key() == Qt::Key::Key_S)
	{
		sDown = true;
	}
	else if(event->key() == Qt::Key::Key_A)
	{
		aDown = true;
	}
	else if(event->key() == Qt::Key::Key_D)
	{
		dDown = true;
	}
	else if(event->key() == Qt::Key::Key_V)
	{
		vDown = true;
	}
}
void QGLRenderer::keyReleaseEvent ( QKeyEvent * event )
{
	if(event->isAutoRepeat())
		return;

	if(event->key() == Qt::Key::Key_W)
	{
		wDown = false;
	}
	else if(event->key() == Qt::Key::Key_S)
	{
		sDown = false;
	}
	else if(event->key() == Qt::Key::Key_A)
	{
		aDown = false;
	}
	else if(event->key() ==  Qt::Key::Key_D)
	{
		dDown = false;
	}
	else if(event->key() == Qt::Key::Key_V)
	{
		vDown = false;
	}
}

void QGLRenderer::navigateUpdate()
{
	float avgGridScale = ( mesh->getSize().x + mesh->getSize().z) /2;
	float scale = 0.5 * avgGridScale/1000;
	scale *= navigationSpeed;

	if (wDown) 
	{
		cam->moveCamera(Camera::FORWARD,moveSpeed * scale );
		transfUpdate = true;
	}
	if(sDown)
	{
		cam->moveCamera(Camera::BACK,moveSpeed * scale);
		transfUpdate = true;
	}
	if(aDown)
	{
		cam->moveCamera(Camera::LEFT,moveSpeed * scale);
		transfUpdate = true;
	}
	if(dDown)
	{
		cam->moveCamera(Camera::RIGHT,moveSpeed * scale);
		transfUpdate = true;
	}
	if(vDown)
	{
		emit sourceButton();
	}
}

void QGLRenderer::mouseMoveEvent ( QMouseEvent * event )
{
	if(mouseDown)
	{
		float rx = (lastx - event->pos().x())*rotateSpeed;
		float ry = (lasty - event->pos().y())*rotateSpeed;

		cam->rotateCamera(rx,ry);
		lastx = event->pos().x();
		lasty = event->pos().y();
		transfUpdate = true;
	}


	GLint vptr[4];
	glGetIntegerv(GL_VIEWPORT,vptr);
	glm::vec4 vp(vptr[0],vptr[1],vptr[2],vptr[3]);
	
	//invert y ( screens zero is at the top left corner, we want it at the bottom left corner)
	int x,y;

	y = vptr[3] - event->pos().y();
	x = event->pos().x();

	//retrieve z-depth from current framebuffer
	float z; 
	glReadPixels(x,y,1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&z);
	
	//reverse viewProjection transformation 
	glm::vec3 mouseCoord = glm::unProject(glm::vec3(x,y,z),
		cam->getCameraTransform() , projMat, vp);

	currentMousePosX = clamp(mouseCoord.x / (mesh->getSize().x-1),0.0f,1.0f);
	currentMousePosZ = clamp(mouseCoord.z / (mesh->getSize().z-1),0.0f,1.0f);

	glUseProgram(layerShader->getProgram());
	glUniform2f(layerShader->getUniformLocation("mousePos"),
		currentMousePosX, currentMousePosZ );

	glUseProgram(waterShader->getProgram());
	glUniform2f(waterShader->getUniformLocation("mousePos"),
		currentMousePosX, currentMousePosZ);

}
void QGLRenderer::mouseReleaseEvent ( QMouseEvent * event )
{
	mouseDown = false;
}

void QGLRenderer::mousePressEvent ( QMouseEvent * event )
{
	lastx = event->pos().x();
	lasty = event->pos().y();	
	mouseDown = true;
}

void QGLRenderer::wheelEvent ( QWheelEvent * event )
{
	emit brushScroll(event->delta());
}

void QGLRenderer::leaveEvent ( QEvent * event )
{
	//ensure down states are correct when widget looses focus
	mouseDown = wDown = sDown = dDown = aDown = vDown = false;
}