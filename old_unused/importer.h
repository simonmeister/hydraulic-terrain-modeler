#ifndef IMPORTER_H
#define IMPORTER_H

#include <Qt/qimage.h>
#include <string>

//HyTM includes
#include "terrain.h"

namespace HyTM
{
	class Importer
	{
	public:
		virtual CPU_Field load() = 0;
	};

	class ImageImporter : public Importer
	{
	public:
		ImageImporter(const std::string& fileName);
		~ImageImporter();

		void setScaleFactor(float scaleX, float scaleY);

		CPU_Field load();
	private:
		std::string fileName;
	};

	class AsciiImporter : public importer
	{
	public:

	};
}
#endif //IMPORTER_H