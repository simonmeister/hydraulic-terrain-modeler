#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include "ui_mainwindow.h"
#include "coreApi.h"
#include "qt-opengl.h"

class mainwindow : public QMainWindow
{
	Q_OBJECT

public:
	mainwindow(QtHyTM * hytm,QGLRenderer* r, QWidget *parent = 0);
	~mainwindow();

private:
	Ui::mainwindow* ui;

	QGLRenderer* renderer;
	QtHyTM* api;

	QTimer* updateGui;

	bool constMarkerVisible;
private slots:
	void onAboutHytm();

	void updateStuff();
	void showTerrainInfo();

	void onSimulationSettings();

	void onRainSettings();
	void onWaterSettings();
	void onErosionSettings();
	void onInteractiveMarkerSettings();
	void onConstantMarkerSettings();


	void onAddSource();
	void onDeleteSource();
	void onDeleteAllSources();
	void onCurrentSourceSettings();
	void onSourceListSelection();

	void onGLSettings();
	void onGLSelectWaterBaseColor();
	void onGLSelectMarker();
	void onGLSelectConstantMarker();
	void onGLSelectBGColor();

	void onTerrainScale();
	void onTerrainNew();
	void onTerrainLoad();
	void onTerrainSave();
	void onTerrainAdd();
	void onTerrainSaveAll();
	void onTerrainDelete();
	void onTerrainMaterialColor();
	void onTerrainMaterialDensity();
	void onResetWater();
	void onLayerListSelection();
	void onCurrentLayerSettings();
	void onCurrentLayerColor();

	void onBrushScroll(int delta);
	void onSourceButton();
};

#endif // MAINWINDOW_H
