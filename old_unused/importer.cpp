#include "importer.h"

#include <fstream>

namespace HyTM
{
	ImageImporter::ImageImporter(const std::string& fn)
		: fileName(fn)
	{
		std::ifstream s(fn.c_str(),std::ios_base::in);
		if (!s.is_open() || !s.good())
		{
			s.close();
			throw "Invalid image file!";
		}
		s.close();	
	}
	CPU_Field ImageImporter::load()
	{
		QImage loadImage;
		loadImage.load(QString::fromStdString(fileName));
		loadImage.convertToFormat(QImage::Format_RGB32);
		unsigned char* origData = loadImage.bits();
		size_t rezZ = loadImage.size().width();
		size_t rezX = loadImage.size().height();

		float* mem = new float(rezX*rezZ);
		for(int i = 0,k=0; i < rezX;i++)
		{
			for(int j = 0; j<rezZ;j++,k+=3)
			{
				mem[i*rezX +j] = (float)origData[k]/255.0;
			}
		}
		
		CPU_Field ff;
		ff.size.x = rezX;
		ff.size.z = rezZ;
		ff.ptr = mem;
		return ff;
	}

}