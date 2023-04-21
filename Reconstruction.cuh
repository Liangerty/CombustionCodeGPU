#pragma once
#include "Define.h"
#include "Field.h"

namespace cfd {
struct DParameter;
class Reconstruction {
public:
  integer limiter=0;

  CUDA_CALLABLE_MEMBER explicit Reconstruction(DParameter* param);

  __device__ virtual void
  apply(real *pv, real *pv_l, real *pv_r, const integer idx_shared, const integer n_var);

  ~Reconstruction() = default;

private:

};

class MUSCL :public Reconstruction {
  constexpr static double kappa{1.0 / 3.0};
public:
  CUDA_CALLABLE_MEMBER explicit MUSCL(DParameter* param);

  __device__ void apply(real *pv, real *pv_l, real *pv_r, const integer idx_shared, const integer n_var) override;
};

__device__ void
reconstruction(real *pv, real *pv_l, real *pv_r, Reconstruction *&method, const integer idx_shared, DZone *zone,
               DParameter *param);
} // cfd
