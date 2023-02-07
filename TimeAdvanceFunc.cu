#include "TimeAdvanceFunc.cuh"
#include "Field.h"
#include "Mesh.h"
#include "InviscidScheme.cuh"

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

void
cfd::compute_inviscid_flux(const Block &block, cfd::DZone *zone, InviscidScheme **inviscid_scheme, DParameter *param,
                           const integer n_var) {
  const integer extent[3]{block.mx, block.my, block.mz};
  const integer ngg{block.ngg};
  const integer dim{extent[2] == 1 ? 2 : 3};
  constexpr integer block_dim=64;
  const integer n_computation_per_block=block_dim+2*ngg-1;
  const auto shared_mem=(block_dim*n_var // fc
                  +n_computation_per_block*(n_var+3+1))* sizeof(real); // pv[n_var]+metric[3]+jacobian

  for (auto dir = 0; dir < dim; ++dir) {
    integer tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    integer bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    inviscid_flux_1d<<<BPG, TPB, shared_mem>>>(zone, inviscid_scheme, dir, extent[dir], param);
  }
}

__global__ void
cfd::inviscid_flux_1d(cfd::DZone *zone, InviscidScheme **inviscid_scheme, integer direction, integer max_extent,
                      DParameter *param) {
  integer labels[3]{0, 0, 0};
  labels[direction] = 1;
  const integer tid = threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2];
  const integer block_dim = blockDim.x * blockDim.y * blockDim.z;
  const auto ngg{zone->ngg};
  const integer n_point = block_dim + 2 * ngg - 1;

  integer idx[3];
  idx[0] = ((integer) blockDim.x - labels[0]) * blockIdx.x + threadIdx.x;
  idx[1] = ((integer) blockDim.y - labels[1]) * blockIdx.y + threadIdx.y;
  idx[2] = ((integer) blockDim.z - labels[2]) * blockIdx.z + threadIdx.z;
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{zone->n_var};
  real *pv = s;
  real *metric = &pv[n_point * n_var];
  real *jac = &metric[n_point * 3];
  real *fc = &jac[n_point];


  const auto n_spec{zone->n_spec};
  //
  const integer i_shared = tid - 1 + ngg;
  for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
    pv[i_shared * n_var + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
  for (auto l = 0; l < n_spec; ++l) { // 5+l - Y_l
    pv[i_shared * n_var + 5 + l] = zone->yk(idx[0], idx[1], idx[2], l);
  }
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const integer g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_var + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_spec; ++l) { // 5+l - Y_l
        pv[ig_shared * n_var + 5 + l] = zone->yk(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const integer g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_var + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_spec; ++l) { // 5+l - Y_l
        pv[ig_shared * n_var + 5 + l] = zone->yk(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

  (*inviscid_scheme)->compute_inviscid_flux(zone, pv, tid, param, fc, metric, jac);
  __syncthreads();

  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}
