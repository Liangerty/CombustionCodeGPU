#include "TimeAdvanceFunc.cuh"
#include "Field.h"

__global__ void cfd::store_last_step(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  integer j = blockDim.y * blockIdx.y + threadIdx.y;
  integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  zone->bv_last(i, j, k, 0) = zone->bv(i, j, k, 0);
  zone->bv_last(i, j, k, 1) = zone->vel(i, j, k);
  zone->bv_last(i, j, k, 2) = zone->bv(i, j, k, 4);
  zone->bv_last(i, j, k, 3) = zone->bv(i, j, k, 5);
}

__global__ void cfd::compute_square_of_dbv(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  integer j = blockDim.y * blockIdx.y + threadIdx.y;
  integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  auto &bv_last = zone->bv_last;

  bv_last(i, j, k, 0) = (bv(i, j, k, 0) - bv_last(i, j, k, 0)) * (bv(i, j, k, 0) - bv_last(i, j, k, 0));
  bv_last(i, j, k, 1) = (zone->vel(i, j, k) - bv_last(i, j, k, 1)) * (zone->vel(i, j, k) - bv_last(i, j, k, 1));
  bv_last(i, j, k, 2) = (bv(i, j, k, 4) - bv_last(i, j, k, 2)) * (bv(i, j, k, 4) - bv_last(i, j, k, 2));
  bv_last(i, j, k, 3) = (bv(i, j, k, 5) - bv_last(i, j, k, 3)) * (bv(i, j, k, 5) - bv_last(i, j, k, 3));
}

