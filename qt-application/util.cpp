#include "util.h"
#include <qerrormessage.h>
#include <qapplication.h>
#include <iostream>

void sendError(const std::string& err, bool crash)
{
	//append to log
	std::cerr << err <<std::endl;

	//show
	QErrorMessage msg;
	msg.setModal(true);
	msg.showMessage(QString::fromStdString(err));
	msg.exec();
	if(crash)
		qFatal("Unable to recover");
}