#include "TimeAdvanceFunc.cuh"
#include "Field.h"

//void cfd::store_last_step(cfd::DZone *zone) {
//  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz}, ng{zone->ngg};
//  const integer num = (mx + 2 * ng) * (my + 2 * ng) * (mz + 2 * ng) * 6;
//
//  cudaMemcpy(zone->bv_last.data(), zone->bv.data(), num * sizeof(real), cudaMemcpyDeviceToDevice);
//}

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

__global__ void cfd::set_dq_to_0(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  integer j = blockDim.y * blockIdx.y + threadIdx.y;
  integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  auto &dq = zone->dq;
  const integer n_var = zone->n_var;
  for (integer l = 0; l < n_var; l++) {
    dq(i, j, k, l) = 0;
  }
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
//  const real d_rho = bv(i, j, k, 0) - bv_last(i, j, k, 0);
//  const real d_u = bv(i, j, k, 1) - bv_last(i, j, k, 1);
//  const real d_v = bv(i, j, k, 2) - bv_last(i, j, k, 2);
//  const real d_w = bv(i, j, k, 3) - bv_last(i, j, k, 3);
//  const real d_p = bv(i, j, k, 4) - bv_last(i, j, k, 4);
//  const real d_t = bv(i, j, k, 5) - bv_last(i, j, k, 5);
//
//  auto& res=zone->residual;
//  res(i, j, k, 0) = d_rho * d_rho;
//  res(i, j, k, 1) = d_u * d_u + d_v * d_v + d_w * d_w;
//  res(i, j, k, 2) = d_p * d_p;
//  res(i, j, k, 3) = d_t * d_t;
}

