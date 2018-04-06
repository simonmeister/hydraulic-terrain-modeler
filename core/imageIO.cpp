#include "core/imageIO.h"

#include "3rdParty/libtif/tiffio.h"
#include "3rdParty/IL/il.h"
#include <iostream>
#include <float.h>
using namespace std;
//// io with a single 32bit tiff using libtif

bool readImageFloat( const std::string& source, float*& destArray, size_2D& destSize)
{
    TIFF* image;
	image = TIFFOpen(source.c_str(), "r");
	if(!image) return false;

	//uint16 photo, bps, spp, fillorder;
	uint16 tmp;
	tsize_t stripSize;
	unsigned long imageOffset, result;
	int stripMax;
	char *buffer;

	// Check that it is of a type that we support
	if((TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &tmp) == 0) || (tmp != 32))
	{
		cerr << "Failed to read image: Type must be 32Bit" << endl;
		return false;
	}

	if((TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &tmp) == 0) || (tmp != 1))
	{
		cerr << "Failed to read image: Channel number must be 1" << endl;
		return false;
	}

	// Read in the possibly multiple strips
	stripSize = TIFFStripSize (image);
	stripMax = TIFFNumberOfStrips (image);
	imageOffset = 0;
  
	if((buffer = (char *) malloc(stripMax * stripSize)) == NULL)
	{
		cerr << "Failed to read image: Image size to big" << endl;
		return false;
	}
	for (int i = 0; i < stripMax; ++i)
	{
		if((result = TIFFReadEncodedStrip (image, i,buffer + imageOffset,
			stripSize)) == -1)
		{
			free(buffer);
			return false;
		}
		imageOffset += result;
	}

	bool err = false;
	if(TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &destSize.z) == 0)
	{
		cerr <<"Failed to read Image: undefined width" << endl;
		err = true;
	}
	if(TIFFGetField(image, TIFFTAG_IMAGELENGTH, &destSize.x) == 0)
	{
		cerr <<"Failed to read Image: undefined height" << endl;
		err = true;
	}

	if(err)
	{
		free(buffer);
		return false;
	}
	destArray = (float*)buffer;

	float* tmpPtr = new float[destSize.x * destSize.z];
	memcpy((void*)tmpPtr,destArray,sizeof(float) * destSize.x * destSize.z);

	//mirror cols vertically 
	for(int i = 0; i< destSize.x; ++i)
	{
		for(int j = 0; j< destSize.z; ++j)
		{
			destArray[i* destSize.z + j] = tmpPtr[(destSize.x - i -1) * destSize.z +j];
		}
	}
	delete[] tmpPtr;
	//scale by min and max
	float min,max;
	if(TIFFGetField(image, TIFFTAG_MAXSAMPLEVALUE, &max) == 0
		&& TIFFGetField(image, TIFFTAG_MINSAMPLEVALUE, &min) == 0)
	{
		float range = abs(max-min);
		if(range > 0.000001)
		{
			for(int i = 0; i< destSize.x; ++i)
			{
				for(int j = 0; j< destSize.z; ++j)
				{
					destArray[i* destSize.z + j] = destArray[i* destSize.z + j] * range;
				}
			} 
		}
	}
	TIFFClose(image);
	return true;
}


bool writeImageFloat( const std::string& target, float* values, size_2D size)
{
	TIFF *image;

	// Open the TIFF file
	if((image = TIFFOpen(target.c_str(), "w")) == NULL)
		return false;

	// We need to set some values for basic tags before we can add any data
	TIFFSetField(image, TIFFTAG_IMAGEWIDTH, size.z);
	TIFFSetField(image, TIFFTAG_IMAGELENGTH, size.x);
	TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 32);
	TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
	TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, size.z);

	TIFFSetField(image, TIFFTAG_COMPRESSION, 1);
	TIFFSetField(image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
	TIFFSetField(image, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
	TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

	TIFFSetField(image, TIFFTAG_XRESOLUTION, size.x);
	TIFFSetField(image, TIFFTAG_YRESOLUTION, size.z);
	TIFFSetField(image, TIFFTAG_RESOLUTIONUNIT, RESUNIT_NONE);

	//min/max pixel values
	float min = FLT_MAX;
	float max = -FLT_MAX;
	for(int i = 0; i< size.x* size.z; ++i)
	{
		if( values[i] > max)
			max = values[i];
		if( values[i] < min)
			min = values[i];
	}
	TIFFSetField(image, TIFFTAG_MAXSAMPLEVALUE, max);
	TIFFSetField(image, TIFFTAG_MINSAMPLEVALUE, min);

	//mirror image
	float* tmpPtr = new float[size.x * size.z];
	memcpy((void*)tmpPtr,values,sizeof(float) * size.x * size.z);
	for(int i = 0; i< size.x; ++i)
	{
		for(int j = 0; j< size.z; ++j)
		{
			tmpPtr[i* size.z + j] = values[(size.x - i -1) * size.z +j];
		}
	}
	


	// Write the information to the file
	size_t offset = 0;
	size_t stripSize = sizeof(float) * size.z;
	for(int i = 0; i < size.x; ++i)
	{
		TIFFWriteEncodedStrip(image,i, ((char*)tmpPtr) + offset,stripSize );
		offset+= stripSize;
	}
	delete[] tmpPtr;


	// Close the file
	TIFFClose(image);
	return true;
}

/* devIL
bool readImageFloat( const std::string& source, float*& destArray, size_2D& destSize)
{
	unsigned int imageID;
	ilGenImages(1, &imageID); 

	ilBindImage(imageID);
	ilEnable(IL_ORIGIN_SET);
	ilOriginFunc(IL_ORIGIN_LOWER_LEFT); 

	ILboolean success;
	success = ilLoadImage((ILstring)source.c_str());
	if (!success) 
	{
		ilDeleteImages(1, &imageID); 	
		return false;
	}

	//convert to 32bit (float) per channel, number of channels : 1
	ilConvertImage(IL_LUMINANCE, IL_FLOAT); 
	destSize.cols = ilGetInteger(IL_IMAGE_WIDTH);
	destSize.rows = ilGetInteger(IL_IMAGE_HEIGHT);
	
	destArray = new float[destSize.rows * destSize.cols];
	memcpy(destArray,ilGetData(), destSize.rows * destSize.cols *sizeof(float));

	ilDeleteImages(1, &imageID); 
	return true;
}


bool writeImageFloat( const std::string& target, float* values, size_2D size)
{
	unsigned int imageID;
	ilGenImages(1, &imageID); 

	ilBindImage(imageID);
	ilEnable(IL_ORIGIN_SET);
	ilOriginFunc(IL_ORIGIN_LOWER_LEFT); 

	ilTexImage(size.cols ,size.rows,1,1
				,IL_LUMINANCE,IL_FLOAT,values);

	ILboolean success;
	ilEnable(IL_FILE_OVERWRITE);
	success = ilSaveImage((ILstring)target.c_str());
	ilDeleteImages(1,&imageID);
	
	if(success)
		return true;
	return false;
}
*/

////////// arbitrary image format using devIL
bool readImageRGB( const std::string& source, float*& destArray, size_2D& destSize)
{
	unsigned int imageID;
	ilGenImages(1, &imageID); 

	ilBindImage(imageID);
	ilEnable(IL_ORIGIN_SET);
	ilOriginFunc(IL_ORIGIN_LOWER_LEFT); 

	ILboolean success;
	success = ilLoadImage((ILstring)source.c_str());
	if (!success) 
	{
		ilDeleteImages(1, &imageID); 	
		return false;
	}

	//convert to 32bit (float) per channel, number of channels : 1
	ilConvertImage(IL_RGB, IL_FLOAT); 
	destSize.cols = ilGetInteger(IL_IMAGE_WIDTH);
	destSize.rows = ilGetInteger(IL_IMAGE_HEIGHT);
	
	destArray = new float[destSize.rows * destSize.cols * 3];
	memcpy(destArray,ilGetData(), destSize.rows * destSize.cols *sizeof(float) * 3);

	ilDeleteImages(1, &imageID); 
	return true;
}