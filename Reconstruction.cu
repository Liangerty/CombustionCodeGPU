#include "Reconstruction.cuh"
#include "DParameter.h"

namespace cfd {
__host__ __device__ Reconstruction::Reconstruction(DParameter *param):limiter{param->limiter} {

}

__host__ __device__ MUSCL::MUSCL(DParameter *param) : Reconstruction(param) {

}
} // cfd