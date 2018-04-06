#include "mainwindow.h"
#include "qfiledialog.h"
#include "qcolordialog.h"
#include "qdialog.h"
#include "qinputdialog.h"
#include "qmessagebox.h"
#include "util\glm.h"
#include "qcolordialog.h"

#include "core/imageIO.h"

glm::vec3 defaultColor;

mainwindow::mainwindow(QtHyTM * hytm, QGLRenderer* r, QWidget *parent )
	: QMainWindow(parent)
	, ui(new Ui::mainwindow)
	, api(hytm)
{
	ui->setupUi(this);
	renderer = r;
	setCentralWidget(r);
	updateGui = new QTimer();
	connect(updateGui,SIGNAL(timeout()),this,SLOT(updateStuff()));
	updateGui->start(50);
	this->setStatusBar( new QStatusBar());

	defaultColor = glm::vec3(1.0,209.0/255.0, 170.0/255.0);
	//////////////////////GUI controls connect
	connect(ui->act_aboutQt,SIGNAL(triggered()),qApp,SLOT(aboutQt()));
	connect(ui->act_aboutHytm,SIGNAL(triggered()),this,SLOT(onAboutHytm()));
	connect(ui->act_exit,SIGNAL(triggered()),qApp,SLOT(quit()));

	connect(ui->set_iterations,SIGNAL(valueChanged(int)),this,SLOT(onSimulationSettings()));
	connect(ui->set_simulationActive,SIGNAL(stateChanged(int)),this,SLOT(onSimulationSettings()));

	//connect(ui->set_rainDensity,SIGNAL(valueChanged(double)),this,SLOT(onRainSettings()));
	//connect(ui->set_rainVolume,SIGNAL(valueChanged(double)),this,SLOT(onRainSettings()));

	connect(ui->set_waterTimestep,SIGNAL(valueChanged(double)),this,SLOT(onWaterSettings()));
	connect(ui->set_waterDry,SIGNAL(valueChanged(double)),this,SLOT(onWaterSettings()));
	connect(ui->set_waterBoundLevel,SIGNAL(valueChanged(double)),this,SLOT(onWaterSettings()));
	connect(ui->set_waterBoundXPos,SIGNAL(currentIndexChanged(int)),this,SLOT(onWaterSettings()));
	connect(ui->set_waterBoundZPos,SIGNAL(currentIndexChanged(int)),this,SLOT(onWaterSettings()));
	connect(ui->set_waterBoundZNeg,SIGNAL(currentIndexChanged(int)),this,SLOT(onWaterSettings()));
	connect(ui->set_waterBoundXNeg,SIGNAL(currentIndexChanged(int)),this,SLOT(onWaterSettings()));
	connect(ui->set_waterActive,SIGNAL(stateChanged(int)),this,SLOT(onWaterSettings()));
	connect(ui->set_dryExit,SIGNAL(stateChanged(int)), this, SLOT( onWaterSettings()));
	
	connect(ui->set_erosionActive,SIGNAL(stateChanged(int)),this,SLOT(onErosionSettings()));
	connect(ui->set_erosionTimestep,SIGNAL(valueChanged(double)),this,SLOT(onErosionSettings()));
	//set Constants for marker erosion

	connect(ui->set_showInteractiveMarker,SIGNAL(stateChanged(int)),this,SLOT(onInteractiveMarkerSettings()));
	connect(ui->set_interactiveHardness,SIGNAL(valueChanged(double)),this,SLOT(onInteractiveMarkerSettings()));
	connect(ui->set_interactiveIntensity,SIGNAL(valueChanged(double)),this,SLOT(onInteractiveMarkerSettings()));
	connect(ui->set_interactiveRadius,SIGNAL(valueChanged(double)),this,SLOT(onInteractiveMarkerSettings()));
	connect(ui->set_interactiveMode,SIGNAL(currentIndexChanged(int)),this,SLOT(onInteractiveMarkerSettings()));

	connect(ui->act_constantNew,SIGNAL(clicked(bool)),this,SLOT(onAddSource()));
	connect(ui->act_constantDelete,SIGNAL(clicked(bool)),this,SLOT(onDeleteSource()));
	connect(ui->act_constantDeleteAll,SIGNAL(clicked(bool)),this,SLOT(onDeleteAllSources()));


	connect(ui->set_showConstantMarker,SIGNAL(stateChanged(int)),this,SLOT(onConstantMarkerSettings()));

	connect(ui->set_constantIntensity,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
	connect(ui->set_constantRadius,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
	connect(ui->set_constantPosZ,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
	connect(ui->set_constantPosX,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
	connect(ui->set_constantActive,SIGNAL(stateChanged(int)),this,SLOT(onCurrentSourceSettings()));
	connect(ui->widget_sourceList, SIGNAL(currentRowChanged(int)), this, SLOT(onSourceListSelection()));

	connect(ui->set_markerColorConstant,SIGNAL(clicked(bool)),this, SLOT(onGLSelectConstantMarker()));
	connect(ui->set_markerColorInteractive,SIGNAL(clicked(bool)),this,SLOT(onGLSelectMarker()));
	connect(ui->set_waterColor,SIGNAL(clicked(bool)),this,SLOT(onGLSelectWaterBaseColor()));
	connect(ui->act_selectBackgroundColor,SIGNAL(clicked(bool)),this,SLOT(onGLSelectBGColor()));

	connect(ui->vsync,SIGNAL(stateChanged(int)),this,SLOT(onGLSettings()));
	connect(ui->set_waterVisible,SIGNAL(stateChanged(int)),this,SLOT(onGLSettings()));
	connect(ui->set_terrainVisible,SIGNAL(stateChanged(int)),this,SLOT(onGLSettings()));
	connect(ui->set_waterTransparencyBase,SIGNAL(valueChanged(double)),this,SLOT(onGLSettings()));
	connect(ui->set_waterTransparencyMult,SIGNAL(valueChanged(double)),this,SLOT(onGLSettings()));
	connect(ui->set_colorMode,SIGNAL(currentIndexChanged(int)),this,SLOT(onGLSettings()));
	connect(ui->set_wireframe,SIGNAL(stateChanged(int)),this,SLOT(onGLSettings()));
	connect(ui->set_navigationSpeed,SIGNAL(valueChanged(double)),this,SLOT(onGLSettings()));
	connect(ui->set_waterTransparency,SIGNAL(stateChanged(int)),this,SLOT(onGLSettings()));
	
	connect(ui->set_terrainMult,SIGNAL(valueChanged(double)),this,SLOT(onGLSettings()));
	connect(ui->gl_ppedge,SIGNAL(valueChanged(int)),this,SLOT(onGLSettings()));
	connect(ui->set_terrainLighting,SIGNAL(currentIndexChanged(int)),this,SLOT(onGLSettings()));
	connect(ui->set_occlusionSamples,SIGNAL(valueChanged(int)),this,SLOT(onGLSettings()));

//	connect(ui->act_terrainScale,SIGNAL(clicked(bool)),this,SLOT(onTerrainScale()));
	connect(ui->act_terrainNew,SIGNAL(clicked(bool)),this,SLOT(onTerrainNew()));
	connect(ui->act_terrainLoad,SIGNAL(clicked(bool)),this,SLOT(onTerrainLoad()));
	connect(ui->act_terrainSaveAll,SIGNAL(clicked(bool)),this,SLOT(onTerrainSaveAll()));
	connect(ui->act_terrainSave,SIGNAL(clicked(bool)),this,SLOT(onTerrainSave()));
	connect(ui->act_terrainDelete,SIGNAL(clicked(bool)),this,SLOT(onTerrainDelete()));
	connect(ui->act_terrainColor,SIGNAL(clicked(bool)),this,SLOT(onTerrainMaterialColor()));
	connect(ui->act_terrainAdd,SIGNAL(clicked(bool)),this,SLOT(onTerrainAdd()));
	connect(ui->set_terrainDensity,SIGNAL(valueChanged(double)),this,SLOT(onTerrainMaterialDensity()));
	connect(ui->act_resetWater,SIGNAL(clicked(bool)),this,SLOT(onResetWater()));

	connect(ui->widget_layerList,SIGNAL(currentRowChanged(int)),this,SLOT(onLayerListSelection()));
	connect(ui->set_terrainDensity,SIGNAL(valueChanged(double)), this, SLOT(onCurrentLayerSettings()));
	connect(ui->act_terrainColor, SIGNAL(clicked(bool)), this, SLOT(onCurrentLayerColor()));

	//glwidget emissions
	connect(renderer,SIGNAL(brushScroll(int)),this,SLOT(onBrushScroll(int)));
	connect(renderer,SIGNAL(sourceButton()),this,SLOT(onSourceButton()));

	//init default values
	onRainSettings();
	onWaterSettings();
	onErosionSettings();
	onInteractiveMarkerSettings();
	onGLSettings();
	onSourceListSelection();
	onLayerListSelection();
	onConstantMarkerSettings();
	onSimulationSettings();
	showTerrainInfo();
}

mainwindow::~mainwindow()
{

}

void mainwindow::onCurrentLayerSettings()
{
	int idx = ui->widget_layerList->currentRow();
	if(idx == -1)
		return;

	api->setMaterialProp(idx,ui->set_terrainDensity->value());
}

void mainwindow::onCurrentLayerColor()
{
	int idx = ui->widget_layerList->currentRow();
	if(idx == -1)
		return;

	Material* mat = api->getMaterial(idx);

	QColor clr = QColorDialog::getColor(QColor(mat->getColor().r*255.0f, mat->getColor().g*255.0f, mat->getColor().b*255.0f));

	api->setMaterialColor(idx,glm::vec3(clr.red()/255.0f, clr.green()/255.0f, clr.blue()/255.0f));
}

void mainwindow::onLayerListSelection()
{
	int idx = ui->widget_layerList->currentRow();
	if(idx == -1)
	{
		ui->set_terrainDensity->setValue(0.0f);
		ui->set_terrainDensity->setEnabled(false);
		ui->act_terrainColor->setEnabled(false);
		ui->act_terrainDelete->setEnabled(false);
	}
	else
	{
		ui->act_terrainDelete->setEnabled(true);
		ui->act_terrainColor->setEnabled(true);
		ui->set_terrainDensity->setEnabled(true);
		Material *mat = api->getMaterial(idx);
		float d = mat->getDensity();
		ui->set_terrainDensity->setValue(d);
	}
}

void mainwindow::onAboutHytm()
{
	QMessageBox msg;
	msg.setModal(true);
	msg.setText("HYdraulic Terrain Modeller. \nSimon Meister, 2013. \nsimon.meister@servomold.de");
	msg.exec();
}

void mainwindow::updateStuff()
{
	QString statText;
	statText.append("FPS: ");
	statText.append(QString::number(renderer->getLastFPS()));
	statText.append("     |     ");
	statText.append("PC: ");
	statText.append(QString::number(renderer->getLastPrimitiveCount()));
	statText.append("     |     ");
	statText.append("RPS: ");
	statText.append(QString::number(api->getActualTicks()));
	statText.append("     |     ");

	if(api->getSimulationActive())
		statText.append("Simulation aktiv");
	else
		statText.append("Simulation angehalten");
	this->statusBar()->showMessage(statText);
}

void mainwindow::onSimulationSettings()
{
	api->setCurrentTicks(ui->set_iterations->value());
	api->setSimulationActive(ui->set_simulationActive->isChecked());
}

void mainwindow::onRainSettings()
{

}


void mainwindow::onWaterSettings()
{
	WaterPipeSimulator* ws = api->getWSim();
	if(!ws)
		return;

	ws->setDryTreshold(ui->set_waterDry->value());
	ws->setTimestep(ui->set_waterTimestep->value());
	ws->setActive(ui->set_waterActive->isChecked());
	ui->wsBox->setChecked(ui->set_waterActive->isChecked());
	ws->setDryOut(ui->set_dryExit->isChecked());

	float level = ui->set_waterBoundLevel->value();

	if(ui->set_waterBoundXPos->currentIndex() == 0)
		ws->setBoundaryReflect(WaterPipeSimulator::BOUND_RIGHT);
	else
		ws->setBoundaryFixed(WaterPipeSimulator::BOUND_RIGHT,level);

	if(ui->set_waterBoundXNeg->currentIndex() == 0)
		ws->setBoundaryReflect(WaterPipeSimulator::BOUND_LEFT);
	else
		ws->setBoundaryFixed(WaterPipeSimulator::BOUND_LEFT,level);

	if(ui->set_waterBoundZPos->currentIndex() == 0)
		ws->setBoundaryReflect(WaterPipeSimulator::BOUND_TOP);
	else
		ws->setBoundaryFixed(WaterPipeSimulator::BOUND_TOP,level);

	if(ui->set_waterBoundZNeg->currentIndex() == 0)
		ws->setBoundaryReflect(WaterPipeSimulator::BOUND_BOTTOM);
	else
		ws->setBoundaryFixed(WaterPipeSimulator::BOUND_BOTTOM,level);
}

void mainwindow::onErosionSettings()
{
	WaterPipeSimulator* ws = api->getWSim();
	if(!ws)
		return;

	ws->erode =ui->set_erosionActive->isChecked();
	ws->erodeconst = ui->set_erosionTimestep->value();
}

void mainwindow::onGLSelectBGColor()
{
	glm::vec3 prev = renderer->getBGColor();
	QColor clr = QColorDialog::getColor(QColor(prev.r*255,prev.g*255,prev.b*255));
	renderer->setBGColor(glm::vec3(clr.red()/255.0f,clr.green()/255.0f,clr.blue()/255.0f));
}

void mainwindow::onInteractiveMarkerSettings()
{
	RadialBrush* rb = api->getBrush();
	rb->setHardness(ui->set_interactiveHardness->value());
	rb->setIntensity(ui->set_interactiveIntensity->value());
	rb->setRadius(ui->set_interactiveRadius->value());
	renderer->setBrushRadius(ui->set_interactiveRadius->value());
	renderer->setShowMarker(ui->set_showInteractiveMarker->isChecked());

	if( ui->set_interactiveMode->currentIndex() == 0)
		rb->setBrushWater();
	else
		rb->setBrushLayer(ui->set_interactiveMode->currentIndex()-1);
}

void mainwindow::onConstantMarkerSettings()
{
	constMarkerVisible = ui->set_showConstantMarker->isChecked();

	//make sure marker is turned off if false
	onSourceListSelection();
}

void mainwindow::onBrushScroll(int delta)
{	
	ui->set_interactiveRadius->setValue(clamp( ui->set_interactiveRadius->value() + delta /50000.0f,0.0,1.0));
	onInteractiveMarkerSettings();
}

void mainwindow::onSourceButton()
{
	api->paint(renderer->getMousePosX(), renderer->getMousePosZ());
}

void mainwindow::onAddSource()
{
	if(api->getSSim() == 0)
		return;

	float r,px,pz,i;
	bool ok;
	px = QInputDialog::getDouble(this,"Einrichten", "Position X", 0.5, 0.0, 1.0,5,&ok);
	if(!ok) return;
	pz = QInputDialog::getDouble(this,"Einrichten", "Position Z", 0.5, 0.0, 1.0,5,&ok);
	if(!ok) return;
	r = QInputDialog::getDouble(this,"Einrichten", "Radius", 0.01, 0.0, 1.0,4,&ok);
	if(!ok) return;
	i = QInputDialog::getDouble(this,"Einrichten", "Intensität", 0.0, -1000.0, 1000.0,4,&ok);
	if(!ok) return;

	api->getSSim()->addRadialSource(px,pz,i,r);
	QString name("Quelle ");
	name.append(QString::number(ui->widget_sourceList->count()));
	ui->widget_sourceList->addItem(name);
}


void mainwindow::onDeleteSource()
{
	int idx = ui->widget_sourceList->currentRow();
	if(idx == -1)
		return;

	delete ui->widget_sourceList->takeItem(idx);
	api->getSSim()->deleteSource(idx);
}


void mainwindow::onDeleteAllSources()
{
	int count = ui->widget_sourceList->count();
	while(count > 0)
	{
		--count;
		delete ui->widget_sourceList->takeItem(0);
		api->getSSim()->deleteSource(0);
	}
	renderer->setShowConstantMarker(false);
}


void mainwindow::onCurrentSourceSettings()
{	


	int idx = ui->widget_sourceList->currentRow();
	if(idx == -1)
		return;

	radialSource* src = (radialSource*) api->getSSim()->getSource(idx);
	src->active = ui->set_constantActive->isChecked();
	src->intensity = ui->set_constantIntensity->value();
	src->nPosX = ui->set_constantPosX->value();
	src->nPosZ = ui->set_constantPosZ->value();
	src->nRadius = ui->set_constantRadius->value();
	api->getSSim()->renewSources();


	renderer->setConstantMarkerPosition(glm::vec2(src->nPosX, src->nPosZ));
	renderer->setConstantMarkerRadius(src->nRadius);
	renderer->setShowConstantMarker(ui->set_constantActive->isChecked() && constMarkerVisible);
}

void mainwindow::onSourceListSelection()
{
	int idx = ui->widget_sourceList->currentRow();
	if(idx == -1)
	{
		ui->set_constantActive->setChecked(0);
		ui->set_constantIntensity->setValue(0.0);
		ui->set_constantPosX->setValue(0.0);
		ui->set_constantPosZ->setValue(0.0);
		ui->set_constantRadius->setValue(0.0);

		ui->set_constantActive->setEnabled(false);
		ui->set_constantIntensity->setEnabled(false);
		ui->set_constantPosX->setEnabled(false);
		ui->set_constantPosZ->setEnabled(false);
		ui->set_constantRadius->setEnabled(false);
		renderer->setShowConstantMarker(false);
		ui->act_constantDelete->setEnabled(false);
	}
	else
	{
		radialSource* src = (radialSource*) api->getSSim()->getSource(idx);

		disconnect(ui->set_constantIntensity,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
		disconnect(ui->set_constantRadius,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
		disconnect(ui->set_constantPosZ,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
		disconnect(ui->set_constantPosX,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
		disconnect(ui->set_constantActive,SIGNAL(stateChanged(int)),this,SLOT(onCurrentSourceSettings()));

		ui->set_constantActive->setChecked(src->active);
		ui->set_constantIntensity->setValue(src->intensity);
		ui->set_constantPosX->setValue(src->nPosX);
		ui->set_constantPosZ->setValue(src->nPosZ);
		ui->set_constantRadius->setValue(src->nRadius);
		renderer->setConstantMarkerPosition(glm::vec2(src->nPosX, src->nPosZ));
		renderer->setConstantMarkerRadius(src->nRadius);
		renderer->setShowConstantMarker(ui->set_constantActive->isChecked() && constMarkerVisible);

		connect(ui->set_constantIntensity,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
		connect(ui->set_constantRadius,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
		connect(ui->set_constantPosZ,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
		connect(ui->set_constantPosX,SIGNAL(valueChanged(double)),this,SLOT(onCurrentSourceSettings()));
		connect(ui->set_constantActive,SIGNAL(stateChanged(int)),this,SLOT(onCurrentSourceSettings()));

		ui->set_constantActive->setEnabled(true);
		ui->set_constantIntensity->setEnabled(true);
		ui->set_constantPosX->setEnabled(true);
		ui->set_constantPosZ->setEnabled(true);
		ui->set_constantRadius->setEnabled(true);
		ui->act_constantDelete->setEnabled(true);
	}
}

void mainwindow::onGLSettings()
{
	renderer->setVsyncOn(ui->vsync->isChecked());
	renderer->setWaterVisible(ui->set_waterVisible->isChecked());
	renderer->setLayersVisible(ui->set_terrainVisible->isChecked());
	renderer->setPixelsPerEdge(ui->gl_ppedge->value());
	renderer->setTransparencyBase(ui->set_waterTransparencyBase->value());
	renderer->setTransparencyMult(ui->set_waterTransparencyMult->value());
	renderer->setTerrainLighting(ui->set_terrainLighting->currentIndex());
	renderer->setTerrainMult(ui->set_terrainMult->value());
	renderer->setOcclusionSamples(ui->set_occlusionSamples->value());
	renderer->setReflections(ui->set_colorMode->currentIndex() == 0);
	renderer->setShaded(!ui->set_wireframe->isChecked());
	renderer->setNavigationSpeed(ui->set_navigationSpeed->value());
	renderer->setTransparency(ui->set_waterTransparency->isChecked());
}


void mainwindow::onGLSelectWaterBaseColor()
{
	glm::vec3 prev = renderer->getWaterBase();
	QColor clr = QColorDialog::getColor(QColor(prev.r*255,prev.g*255,prev.b*255));
	renderer->setWaterBase(glm::vec3(clr.red()/255.0f,clr.green()/255.0f,clr.blue()/255.0f));
}

void mainwindow::onGLSelectMarker()
{
	glm::vec3 prev = renderer->getMarkerColor();
	QColor clr = QColorDialog::getColor(QColor(prev.r*255,prev.g*255,prev.b*255));
	renderer->setMarkerColor(glm::vec3(clr.red()/255.0f,clr.green()/255.0f,clr.blue()/255.0f));
}

void mainwindow::onGLSelectConstantMarker()
{
	glm::vec3 prev = renderer->getConstantMarkerColor();
	QColor clr = QColorDialog::getColor(QColor(prev.r*255,prev.g*255,prev.b*255));
	renderer->setConstantMarkerColor(glm::vec3(clr.red()/255.0f,clr.green()/255.0f,clr.blue()/255.0f));
}

void mainwindow::onTerrainScale()
{
	showTerrainInfo();
}


void mainwindow::onTerrainNew()
{
	if(api->getTerrainLoaded())
	{
		api->addEmptyLayer(api->getRez(),defaultColor,0.1);

		QString name("Schicht ");
		name.append(QString::number(ui->widget_sourceList->count()+1));
		ui->widget_layerList->addItem(name);
	}
	else
	{
		size_2D dim;
		bool ok;
		dim.x = QInputDialog::getInt(this,"Einrichten","Höhe",100,1,10000,1,&ok);
		if(!ok)
			return;
		dim.z = QInputDialog::getInt(this,"Einrichten","Breite",dim.x,1,10000,1,&ok);
		if(!ok)
			return;
		api->addEmptyLayer(dim,defaultColor,0.01);

		QString name("Schicht ");
		name.append(QString::number(ui->widget_sourceList->count()+1));
		ui->widget_layerList->addItem(name);
	}
	showTerrainInfo();
}

void mainwindow::onResetWater()
{
	api->resetWater();
}

void mainwindow::onTerrainLoad()
{

	// query scale --------------------------- 
	bool ok;
	float scale = QInputDialog::getDouble(this,"Einrichten","Höhen-Skalierung",1.0f,0.0f,10000.0f,5,&ok);
	if(!ok)
		return;
	float*src; 
	size_2D size;

	// Load file --------------------------- 
	
	QString fileName = QFileDialog::getOpenFileName(this, tr("Terrain laden"),"../",
                           tr("32Bit-Tiff (*.tiff *.tif)"));

	if (!readImageFloat(fileName.toStdString(),src,size))
	{
		QMessageBox::warning(this,"Fehler beim Laden","Datei nicht vorhanden oder Fehlerhaft.");
		return;
	}

	if(fileName.isEmpty())
		return;

	if(ui->widget_layerList->count() > 0 && (api->getRez().x != size.x || api->getRez().z != size.z))
	{
		QMessageBox::warning(this,"Fehler","Bilddatei konnte nicht geladen werden.\nGröße stimmt nicht überein");
		return;
	}

	api->addLayer(size,src,scale,defaultColor,0.01);
	delete[] src;
	QString name("Schicht ");
	name.append(QString::number(ui->widget_layerList->count()+1));
	ui->widget_layerList->addItem(name);

	showTerrainInfo();
}


void mainwindow::onTerrainSave()
{
	QString fileName = QFileDialog::getSaveFileName(this, tr("Schicht speichern"),"../",
                           tr("32Bit Graustufenbild (*.tif *.tiff)"));

	if(fileName.isEmpty())
		return;
	
	//query save layer ----------------
	QStringList l;
	l.append("Wasser");
	for(int i = 0; i < ui->widget_layerList->count(); ++i)
	{
		QString name("Schicht ");
		name.append(QString::number(i+1));
		l.append(name);
	}
	bool ok;
	QString result = QInputDialog::getItem(this,"Konfigurieren","Schicht",l,0,false,&ok);
	if(!ok) return;

	float* dataField;
	size_2D size;

	if(result.compare("Wasser") == 0)
	{
		dataField = api->writeWater(size);
	}
	//get index of result
	else
	{
		unsigned int id;
		for(unsigned int i = 0; i<l.size(); ++i)
		{
			if(l[i].compare(result) == 0)
			{
				id = i;
				break;
			}
		}
		dataField = api->writeLayer(size,id-1);
	}

	//write if possible----------------
	ok = writeImageFloat(fileName.toStdString(), dataField, size);

	if(!ok)
		QMessageBox::warning(this,"Fehler beim schreiben","Datei konnte nicht gespeichert werden.");
}


void mainwindow::onTerrainSaveAll()
{
	QString fileName = QFileDialog::getSaveFileName(this, tr("Schicht speichern"),"../",
                           tr("Bilddatei (*.bmp *.jpg *.exr .*tga)"));

	if(fileName.isEmpty())
		return;
	//query save mode ----------------
	QStringList l;
	l.append("Wasserhöhen auslassen");
	l.append("Wasserhöhen hinzufügen");
	bool ok;
	QString result = QInputDialog::getItem(this,"Konfigurieren","Wasser als Ebene aufaddieren?",l,0,false,&ok);
	if(!ok) return;

	bool ww;
	if(result.compare("Wasserhöhen auslassen") == 0)
		ww = false;
	else
		ww = true;

	//write if possible save mode ----------------
	size_2D size;
	float* dataField = api->write(size,ww);
	ok = writeImageFloat(fileName.toStdString(), dataField, size);

	if(!ok)
		QMessageBox::warning(this,"Fehler beim schreiben","Datei konnte nicht gespeichert werden.");
}

void mainwindow::onTerrainDelete()
{
	int idx = ui->widget_layerList->currentRow();
	if(idx == -1)
		return;

	delete ui->widget_layerList->takeItem(idx);
	api->removeLayer(idx);

	showTerrainInfo();
}

void mainwindow::onTerrainMaterialColor()
{

}

void mainwindow::onTerrainMaterialDensity()
{

}

void mainwindow::onTerrainAdd()
{
	// query scale --------------------------- 
	bool ok;
	float scale = QInputDialog::getDouble(this,"Einrichten","Höhen-Skalierung",1.0f,0.0f,10000.0f,1,&ok);
	if(!ok)
		return;

	float*src; 
	size_2D size;

	// Load file --------------------------- 
	QString fileName = QFileDialog::getOpenFileName(this, tr("Terrain laden"),"../",
                           tr("Bilddatei (*.tif *.tiff)"));

	if(fileName.isEmpty())
		return;

	if (!readImageFloat(fileName.toStdString(),src,size))
	{
		QMessageBox::warning(this,"Fehler beim Laden","Datei nicht vorhanden oder Fehlerhaft.");
		return;
	}

	// query layer --------------------------- 
	QStringList list;
	list.append("Wasser");
	for(int i = 0; i < ui->widget_layerList->count(); ++i)
	{
		QString name("Schicht ");
		name.append(QString::number(i+1));
		list.append(name);
	}
	QString result = QInputDialog::getItem(this,"Schicht auswählen", "",list,0,false,&ok);
	if(!ok)
		return;

	if(result.compare("Wasser") == 0)
	{
		api->addWater(src,size,scale);
	}
	else
	{
		unsigned int id;
		for(unsigned int i = 0; i<list.size(); ++i)
		{
			if(list[i].compare(result) == 0)
			{
				id = i;
				break;
			}
		}
		api->addLayerHeight(src,size,id-1,scale);
	}
}

void mainwindow::showTerrainInfo()
{
	//update dimension info
	if(ui->widget_layerList->count() > 0)
	{
		ui->show_terrainResX->setText(QString::number(api->getRez().x));
		ui->show_terrainResZ->setText(QString::number(api->getRez().z));
		ui->show_terrainAspectRatio->setText(QString::number(api->getRez().x/api->getRez().z));

		ui->act_terrainAdd->setEnabled(true);
		ui->act_terrainSave->setEnabled(true);
		ui->act_terrainSaveAll->setEnabled(true);
//		ui->act_terrainScale->setEnabled(true);
		ui->act_resetWater->setEnabled(true);

		onWaterSettings();
		onErosionSettings();
		onRainSettings();
	}
	else
	{
		ui->act_resetWater->setEnabled(false);
	//	ui->act_terrainScale->setEnabled(false);
		ui->act_terrainSave->setEnabled(false);
		ui->act_terrainSaveAll->setEnabled(false);
		ui->act_terrainAdd->setEnabled(false);
		ui->show_terrainResX->setText("");
		ui->show_terrainResZ->setText("");
		ui->show_terrainAspectRatio->setText("");
	}
	//update comboboxes
	ui->set_interactiveMode->clear();
	ui->set_interactiveMode->addItem("Wasser");
	for(int i = 0; i < ui->widget_layerList->count(); ++i)
	{
		QString name("Schicht ");
		name.append(QString::number(i+1));
		ui->set_interactiveMode->addItem(name);
	}
}