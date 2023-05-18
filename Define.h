#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#define __host__
#endif

using integer = int;
using real = double;
using uint = unsigned int;
