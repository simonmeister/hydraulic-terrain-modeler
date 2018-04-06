#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <string>

#include "util\global.h"

bool readImageFloat( const std::string& source, float*& destArray, size_2D& destSize);

bool readImageRGB( const std::string& source, float*& destArray, size_2D& destSize);

bool writeImageFloat( const std::string& target, float* values, size_2D size);

#endif IMAGE_IO_H
	
