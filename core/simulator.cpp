#include "simulator.h"

#include <cuda_runtime.h>
#include "3rdParty\cuda-helper\helper_cuda.h"

//------------------------------------------------------------------------
// Resource Manager definitions
//------------------------------------------------------------------------
/*
void SharedCudaResources::register2DResource(const std::string& name, size_2D size, bool p)
{
	simResource r;
	r.permanent = p;
	r.size = size;

	checkCudaErrors(cudaMallocPitch(&r.field.devPtr, &r.field.pitch,
					size.cols * sizeof(float), size.rows));
	resources2D.insert(std::pair<std::string, simResource>(name, r));
}

void SharedCudaResources::unregister2DResource(const std::string& name)
{
	checkCudaErrors(cudaFree( resources2D[name].field.devPtr ));
	resources2D.erase(name);
}

void SharedCudaResources::clearNonPermanent()
{
	for(auto iter = resources2D.begin(); iter != resources2D.end(); ++iter)
	{
		//test That!
		if(iter->second.permanent)
			iter = resources2D.erase(iter);
	}
}
*/