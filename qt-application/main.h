
//hytm
#include "coreApi.h"
#include "qt-opengl.h"
#include "3rdParty\glew\glew.h"
#include "3rdParty\IL\il.h"


#include <iostream>
#include <sstream>
#include <fstream>


//qt
#include "mainwindow.h"
#include <QtGui/QApplication>

class HyTMApplication : public QApplication
{
	Q_OBJECT
public:
	HyTMApplication(int argc, char**argv);

	void initialize();
					
private slots:
	void cleanUp();

private:
	QtHyTM* hytm;
	std::stringstream errBuff;
	QGLRenderer* render;
};