#pragma once
#include "Define.h"

namespace cfd {
struct DParameter;
class Reconstruction {
  integer limiter=0;
public:
  CUDA_CALLABLE_MEMBER explicit Reconstruction(DParameter* param);

  ~Reconstruction() = default;

private:

};

class MUSCL :public Reconstruction {
  constexpr static double kappa{1.0 / 3.0};
public:
  CUDA_CALLABLE_MEMBER explicit MUSCL(DParameter* param);
};
} // cfd
