#include "main.h"
#include "util.h"
using namespace std;

HyTMApplication::HyTMApplication(int argc, char**argv)
:QApplication(argc,argv)
{}

void HyTMApplication::initialize()
{
	this->setApplicationName("HyTM");
	this->setApplicationVersion("0.1");

	//redirect cerr
	std::cerr.rdbuf(errBuff.rdbuf());

	//check for cuda driver and hardware
	cudaError_t err;
	int cnt;
	err = cudaGetDeviceCount(&cnt);
	if(err == cudaErrorNoDevice)
		sendError("Cuda is unsupported by this system.",true);
	else if(err == cudaErrorInsufficientDriver)
		sendError("Unable to find Cuda device driver.",true);

	//check for cuda version
	cudaDeviceProp prop;
	cudaGetDevice(&cnt);
	cudaGetDeviceProperties(&prop,cnt);
	if(prop.major < 3)
		sendError("Cuda version 3.0 or higher is requiered.",true);

	ilInit();

	render = new QGLRenderer();
	hytm = new QtHyTM(render);

	mainwindow* mwin = new mainwindow(hytm, render,0);
	mwin->show();
//	mwin->resize(2000,2000);

	connect(qApp,SIGNAL(aboutToQuit()),this,SLOT(cleanUp()));
}


void HyTMApplication::cleanUp()
{
	//write error log
	std::string text = errBuff.str();
	std::ofstream errstr("log.txt");
	errstr << text;
	errstr.close();

	delete hytm;
	delete render;

	cudaDeviceReset();
}


int main(int argc, char *argv[])
{
	HyTMApplication a(argc, argv);
	a.setStyle("Cleanlooks");
	a.initialize();

	return a.exec();
}
