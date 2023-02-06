#pragma once
#include "Define.h"

namespace cfd{
struct DParameter;
struct DZone;

__device__ void compute_enthalpy(real t, real *enthalpy, DParameter* param);

__device__ void compute_cp(real t, real *cp, DParameter* param);

__device__ void compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, DParameter* param);

__device__ void compute_total_energy(integer i, integer j, integer k, DZone *zone, DParameter* param);

}