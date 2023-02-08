#pragma once
#include "Define.h"

namespace cfd {

struct DZone;
struct DParameter;

class ViscousScheme {
public:
  __device__ ViscousScheme()=default;

  __device__ virtual void compute_fv(integer idx[3], DZone *zone, real *fv, DParameter *param);
  __device__ virtual void compute_gv(integer idx[3], DZone *zone, real *gv, DParameter *param);
  __device__ virtual void compute_hv(integer idx[3], DZone *zone, real *hv, DParameter *param);
};

class SecOrdViscScheme:public ViscousScheme{
public:
  __device__ SecOrdViscScheme();

  __device__ void compute_fv(integer idx[3], DZone *zone, real *fv, cfd::DParameter *param) override;
  __device__ void compute_gv(integer idx[3], DZone *zone, real *gv, cfd::DParameter *param) override;
  __device__ void compute_hv(integer idx[3], DZone *zone, real *hv, cfd::DParameter *param) override;
};

} // cfd
