#ifndef GLOBAL_H
#define GLOBAL_H

const float DEFAULT_TIMESTEP = 0.1f;

///////////////////////////////////////////////////////////////////////
// General macros
#define SAFE_DELETE( X ) \
	if( X ){ delete X; X = nullptr; }

#define SAFE_DELETE_ARRAY( X ) \
	if( X ){ delete[] X; X = nullptr; }


///////////////////////////////////////////////////////////////////////
// Debugging macros
#ifdef _DEBUG
#define IS_DEBUG
#endif

#ifdef IS_DEBUG
#include <assert.h>
#else
#undef assert
#define assert( X )
#endif
//////////////////////////
// general types
union size_2D
{
	size_2D( size_t DIMX, size_t DIMZ)
	{
		rows = DIMX;
		cols = DIMZ;
	}
	size_2D()
	{}
	struct 
	{
		size_t rows; // "height"
		size_t cols; // "width"
	};
	struct
	{
		size_t height; // "height"
		size_t width; // "width"
	};
	struct
	{
		size_t x; // "height"
		size_t z; // "width"
	};
};



#endif //GLOBAL_H