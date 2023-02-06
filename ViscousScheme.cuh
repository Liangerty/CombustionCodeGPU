#pragma once
//#include "Define.h"

namespace cfd {

class ViscousScheme {
public:
  __device__ ViscousScheme()=default;
};

class SecOrdViscScheme:public ViscousScheme{
public:
  __device__ SecOrdViscScheme();
};

} // cfd
