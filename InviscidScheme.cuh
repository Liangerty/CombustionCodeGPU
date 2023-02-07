#pragma once
#include "Define.h"
#include "Reconstruction.cuh"

namespace cfd {
struct DParameter;
struct DZone;

class InviscidScheme {
public:
  Reconstruction* reconstruction_method;

  __device__ explicit InviscidScheme(DParameter* param);

  __device__ virtual void
  compute_inviscid_flux(DZone *zone, real *pv, const integer tid, DParameter *param, real *fc, real *metric,
                        real *jac) =0;

  ~InviscidScheme()=default;
};

class AUSMP:public InviscidScheme{
  constexpr static real alpha{3/16.0}, beta{1/8.0};
public:
  __device__ explicit AUSMP(DParameter* param);

  __device__ void
  compute_inviscid_flux(DZone *zone, real *pv, const integer tid, DParameter *param, real *fc, real *metric,
                        real *jac) override;

  ~AUSMP()=default;
};
} // cfd
