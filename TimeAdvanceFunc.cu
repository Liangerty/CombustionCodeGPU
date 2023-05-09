#include "TimeAdvanceFunc.cuh"
#include "Field.h"
#include "Mesh.h"
#include "InviscidScheme.cuh"
#include "ViscousScheme.cuh"
#include "TemporalScheme.cuh"
#if MULTISPECIES==1
#include "Thermo.cuh"
#else
#include "Constants.h"
#endif

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

__global__ void cfd::set_dq_to_0(cfd::DZone* zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  integer j = blockDim.y * blockIdx.y + threadIdx.y;
  integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  auto& dq = zone->dq;
  const integer n_var = zone->n_var;
  for (integer l = 0; l < n_var; l++)
    dq(i, j, k, l) = 0;
}

void
cfd::compute_inviscid_flux(const Block &block, cfd::DZone *zone, InviscidScheme **inviscid_scheme, DParameter *param,
                           const integer n_var) {
  const integer extent[3]{block.mx, block.my, block.mz};
  const integer ngg{block.ngg};
  const integer dim{extent[2] == 1 ? 2 : 3};
  constexpr integer block_dim=128;
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
  idx[0] = (integer)(( blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = (integer)(( blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = (integer)(( blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{zone->n_var};
  real *pv = s;
  real *metric = &pv[n_point * n_var];
  real *jac = &metric[n_point * 3];
  real *fc = &jac[n_point];


  //
  const integer i_shared = tid - 1 + ngg;
  for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
    pv[i_shared * n_var + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
#if MULTISPECIES==1
  const auto n_spec{zone->n_spec};
  for (auto l = 0; l < n_spec; ++l) { // 5+l - Y_l
    pv[i_shared * n_var + 5 + l] = zone->yk(idx[0], idx[1], idx[2], l);
  }
#endif
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
#if MULTISPECIES==1
      for (auto l = 0; l < n_spec; ++l) { // 5+l - Y_l
        pv[ig_shared * n_var + 5 + l] = zone->yk(g_idx[0], g_idx[1], g_idx[2], l);
      }
#endif
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
#if MULTISPECIES==1
      for (auto l = 0; l < n_spec; ++l) { // 5+l - Y_l
        pv[ig_shared * n_var + 5 + l] = zone->yk(g_idx[0], g_idx[1], g_idx[2], l);
      }
#endif
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

void cfd::compute_viscous_flux(const cfd::Block &block, cfd::DZone *zone, cfd::ViscousScheme **viscous_scheme,
                               cfd::DParameter *param, integer n_var) {
  const integer extent[3]{block.mx, block.my, block.mz};
  const integer dim{extent[2] == 1 ? 2 : 3};
  constexpr integer block_dim=64;

  dim3 tpb{block_dim,1,1};
  dim3 bpg((extent[0]-1)/(block_dim-1)+1,extent[1],extent[2]);
  auto shared_mem=block_dim*n_var* sizeof(real);
  viscous_flux_fv<<<bpg, tpb, shared_mem>>>(zone, viscous_scheme, extent[0], param);

  tpb={1,block_dim,1};
  bpg=dim3(extent[0],(extent[1]-1)/(block_dim-1)+1,extent[2]);
  viscous_flux_gv<<<bpg, tpb, shared_mem>>>(zone, viscous_scheme, extent[1], param);

  if (dim==3){
    tpb={1,1,block_dim};
    bpg= dim3(extent[0],extent[1],(extent[2]-1)/(block_dim-1)+1);
    viscous_flux_hv<<<bpg, tpb, shared_mem>>>(zone, viscous_scheme, extent[2], param);
  }
}

__global__ void
cfd::viscous_flux_fv(cfd::DZone *zone, cfd::ViscousScheme **viscous_scheme, integer max_extent,
                     cfd::DParameter *param) {
  integer idx[3];
  idx[0] = ((integer) blockDim.x - 1) * blockIdx.x + threadIdx.x-1;
  idx[1] = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  idx[2] = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (idx[0] >= max_extent) return;
  const auto tid=threadIdx.x;
  const auto n_var{zone->n_var};

  extern __shared__ real s[];
  real* fv=s;

  (*viscous_scheme)->compute_fv(idx,zone,&fv[tid*n_var],param);
  __syncthreads();

  if (tid>0){
    for (integer l=0;l<n_var;++l)
      zone->dq(idx[0],idx[1],idx[2],l)+=fv[tid*n_var+l]-fv[(tid-1)*n_var+l];
  }
}

__global__ void cfd::viscous_flux_gv(cfd::DZone *zone, cfd::ViscousScheme **viscous_scheme, integer max_extent,
                                     cfd::DParameter *param) {
  integer idx[3];
  idx[0] = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  idx[1] = (integer)((blockDim.y - 1) * blockIdx.y + threadIdx.y)-1;
  idx[2] = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (idx[1] >= max_extent) return;
  const auto tid=threadIdx.y;
  const auto n_var{zone->n_var};

  extern __shared__ real s[];
  real* gv=s;

  (*viscous_scheme)->compute_gv(idx,zone,&gv[tid*n_var],param);
  __syncthreads();

  if (tid>0){
    for (integer l=0;l<n_var;++l)
      zone->dq(idx[0],idx[1],idx[2],l)+=gv[tid*n_var+l]-gv[(tid-1)*n_var+l];
  }
}

__global__ void cfd::viscous_flux_hv(cfd::DZone *zone, cfd::ViscousScheme **viscous_scheme, integer max_extent,
                                     cfd::DParameter *param) {
  integer idx[3];
  idx[0] = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  idx[1] = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  idx[2] = (integer)((blockDim.z - 1) * blockIdx.z + threadIdx.z)-1;
  if (idx[2] >= max_extent) return;
  const auto tid=threadIdx.z;
  const auto n_var{zone->n_var};

  extern __shared__ real s[];
  real* hv=s;

  (*viscous_scheme)->compute_hv(idx,zone,&hv[tid*n_var],param);
  __syncthreads();

  if (tid>0){
    for (integer l=0;l<n_var;++l)
      zone->dq(idx[0],idx[1],idx[2],l)+=hv[tid*n_var+l]-hv[(tid-1)*n_var+l];
  }
}

__global__ void cfd::local_time_step(cfd::DZone *zone, cfd::DParameter *param, TemporalScheme **temporal_scheme) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const integer i=(blockDim.x*blockIdx.x+threadIdx.x);//(integer)
  const integer j=(blockDim.y*blockIdx.y+threadIdx.y);//(integer)
  const integer k=(blockDim.z*blockIdx.z+threadIdx.z);//(integer)
  if (i>=extent[0]||j>=extent[1]||k>=extent[2]) return;

//  (*temporal_scheme)->compute_time_step(zone, i, j, k, param);
  const auto& m{zone->metric(i, j, k)};
  const auto& bv=zone->bv;
  const integer dim{zone->mz == 1 ? 2 : 3};

  const real grad_xi = std::sqrt(m(1, 1) * m(1, 1) + m(1, 2) * m(1, 2) + m(1, 3) * m(1, 3));
  const real grad_eta = std::sqrt(m(2, 1) * m(2, 1) + m(2, 2) * m(2, 2) + m(2, 3) * m(2, 3));
  const real grad_zeta = std::sqrt(m(3, 1) * m(3, 1) + m(3, 2) * m(3, 2) + m(3, 3) * m(3, 3));

  const real u{bv(i, j, k, 1)}, v{bv(i, j, k, 2)}, w{bv(i, j, k, 3)};
  const real U = u * m(1, 1) + v * m(1, 2) + w * m(1, 3);
  const real V = u * m(2, 1) + v * m(2, 2) + w * m(2, 3);
  const real W = u * m(3, 1) + v * m(3, 2) + w * m(3, 3);

  const auto acoustic_speed = zone->acoustic_speed(i, j, k);
  real spectral_radius_inviscid = std::abs(U) + std::abs(V) + acoustic_speed * (grad_xi + grad_eta);
  if (dim == 3)
    spectral_radius_inviscid += std::abs(W) + acoustic_speed * grad_zeta;

  // Next, compute the viscous spectral radius
#if MULTISPECIES==1
  const real coeff_1 = max(zone->gamma(i, j, k), 4.0 / 3.0);
#else
  const real coeff_1 = max(gamma_air, 4.0 / 3.0);
#endif
  const real coeff_2 = zone->mul(i, j, k) / bv(i, j, k, 0) / param->Pr;
  real spectral_radius_viscous = grad_xi * grad_xi + grad_eta * grad_eta;
  if (dim == 3)
    spectral_radius_viscous += grad_zeta * grad_zeta;
  spectral_radius_viscous *= coeff_1 * coeff_2;

  zone->dt_local(i, j, k) = param->cfl / (spectral_radius_inviscid + spectral_radius_viscous);
}

__global__ void cfd::update_cv_and_bv(cfd::DZone *zone, cfd::DParameter *param) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const auto i=(integer)(blockDim.x*blockIdx.x+threadIdx.x);
  const auto j=(integer)(blockDim.y*blockIdx.y+threadIdx.y);
  const auto k=(integer)(blockDim.z*blockIdx.z+threadIdx.z);
  if (i>=extent[0]||j>=extent[1]||k>=extent[2]) return;

  auto& cv=zone->cv;

  real dt_div_jac=zone->dt_local(i,j,k)/zone->jac(i,j,k);
  for (integer l=0;l<zone->n_var;++l)
    cv(i,j,k,l)+=zone->dq(i,j,k,l)*dt_div_jac;
  if (extent[2]==1)
    cv(i,j,k,3)=0;

  auto& bv=zone->bv;
  auto& velocity=zone->vel(i, j, k);

  bv(i, j, k, 0)  = cv(i, j, k, 0);
  const real density_inv = 1.0 / cv(i, j, k, 0);
  bv(i, j, k, 1)  = cv(i, j, k, 1) * density_inv;
  bv(i, j, k, 2)  = cv(i, j, k, 2) * density_inv;
  bv(i, j, k, 3)  = cv(i, j, k, 3) * density_inv;
  velocity = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3); //V^2
#if MULTISPECIES==1
  auto &Y=zone->yk;
  for (int l = 0; l < zone->n_spec; ++l)
    Y(i, j, k, l) = cv(i, j, k, 5 + l) * density_inv;
  compute_temperature(i, j, k, param, zone);
#else
  // Air
  bv(i, j, k, 4) = (gamma_air - 1) * (cv(i, j, k, 4) - 0.5 * bv(i, j, k, 0) * velocity);
  bv(i, j, k, 5) = bv(i, j, k, 4) * mw_air * density_inv / R_u;
#endif
  velocity = std::sqrt(velocity);
}

__global__ void cfd::compute_square_of_dbv(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  integer j = blockDim.y * blockIdx.y + threadIdx.y;
  integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  zone->bv_last(i, j, k, 0) = (zone->bv(i, j, k, 0)-zone->bv_last(i,j,k,0))*(zone->bv(i, j, k, 0)-zone->bv_last(i,j,k,0));
  zone->bv_last(i, j, k, 1) = (zone->vel(i, j, k)-zone->bv_last(i,j,k,1))*(zone->vel(i, j, k)-zone->bv_last(i,j,k,1));
  zone->bv_last(i, j, k, 2) = (zone->bv(i, j, k, 4)-zone->bv_last(i,j,k,2))*(zone->bv(i, j, k, 4)-zone->bv_last(i,j,k,2));
  zone->bv_last(i, j, k, 3) = (zone->bv(i, j, k, 5)-zone->bv_last(i,j,k,3))*(zone->bv(i, j, k, 5)-zone->bv_last(i,j,k,3));
}
