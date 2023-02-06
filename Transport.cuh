#pragma once
#include "Define.h"
#if MULTISPECIES == 1
#include "ChemData.h"
#endif
#ifdef CUDACC
#include <cuda_runtime.h>
#endif
namespace cfd {
__host__ __device__
real Sutherland(real temperature);

#if MULTISPECIES == 1

real compute_viscosity(real temperature, real mw_total, real const *Y, Species &spec);

struct DParameter;
struct DZone;
__device__ void compute_transport_property(integer i, integer j, integer k, real temperature, real mw_total, const real *cp, DParameter* param, DZone* zone);

#endif
}