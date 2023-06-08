#pragma once

#include "Define.h"
#include "Field.h"
#include "DParameter.h"
#include "Constants.h"
#include "Thermo.cuh"

namespace cfd {
struct DZone;

template<MixtureModel mixture_model, TurbMethod turb_method>
__global__ void compute_DQ_0(DZone *zone) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const integer i = blockDim.x * blockIdx.x + threadIdx.x;
  const integer j = blockDim.y * blockIdx.y + threadIdx.y;
  const integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const real dt_local = zone->dt_local(i, j, k);
  const auto &inviscid_spectral_radius = zone->inv_spectr_rad(i, j, k);
  const real diag =
      1 + dt_local * (inviscid_spectral_radius[0] + inviscid_spectral_radius[1] + inviscid_spectral_radius[2]);
  auto &dq = zone->dq;
  const integer n_spec{zone->n_spec};
  if constexpr (mixture_model == MixtureModel::Air || mixture_model == MixtureModel::Mixture) {
    for (integer l = 0; l < 5 + n_spec; ++l) {
      dq(i, j, k, l) /= diag;
    }
  } else if (mixture_model == MixtureModel::FR) {
    // Use point implicit method to dispose chemical source
    for (integer l = 0; l < 5; ++l) {
      dq(i, j, k, l) /= diag;
    }
    // Point implicit
  } // Flamelet method can be added later

  if constexpr (turb_method == TurbMethod::RANS) {
    // switch RANS model, and apply point implicit to treat the turbulent part
  }
}

template<MixtureModel mixture_model>
__device__ void
compute_jacobian_times_dq(const DParameter *param, DZone *zone, const integer i, const integer j, const integer k,
                          const integer dir, real pm_spectral_radius, real *convJacTimesDq) {
  const auto &m = zone->metric(i, j, k);
  const auto xi_x{m(dir + 1, 1)}, xi_y{m(dir + 1, 2)}, xi_z{m(dir + 1, 3)};

  const auto &pv = zone->bv;
  const real u = pv(i, j, k, 1), v = pv(i, j, k, 2), w = pv(i, j, k, 3);
  const real U = xi_x * u + xi_y * v + xi_z * w;
  const real lmd1 = U + pm_spectral_radius;
  const real e = 0.5 * zone->vel(i, j, k) * zone->vel(i, j, k);
  real gamma{gamma_air};
  real b3{0}, b4{0}, h{0};
  auto &dq = zone->dq;
  const auto &sv = zone->sv;

  if constexpr (mixture_model != MixtureModel::Air) {
    const auto &mw = param->mw;
    real enthalpy[MAX_SPEC_NUMBER];
    const real t{pv(i, j, k, 5)};
    compute_enthalpy(t, enthalpy, param);
    gamma = zone->gamma(i, j, k);
    for (int l = 0; l < zone->n_spec; ++l) {
      b3 += R_u * t / mw[l] * dq(i, j, k, 5 + l);
      b4 += enthalpy[l] * dq(i, j, k, 5 + l);
      h += sv(i, j, k, l) * enthalpy[l];
    }
    b3 *= gamma;
    b4 *= gamma - 1;
    h += e;
  } else {
    h = gamma / (gamma - 1) * pv(i, j, k, 4) / pv(i, j, k, 0) + e;
  }
  const double b1 = xi_x * dq(i, j, k, 1) + xi_y * dq(i, j, k, 2) + xi_z * dq(i, j, k, 3) - U * dq(i, j, k, 0);
  const double b2 = (gamma - 1) * (e * dq(i, j, k, 0) - u * dq(i, j, k, 1) - v * dq(i, j, k, 2) - w * dq(i, j, k, 3) +
                                   dq(i, j, k, 4));

  convJacTimesDq[0] = b1 + lmd1 * dq(i, j, k, 0);
  convJacTimesDq[1] = u * b1 + xi_x * b2 + lmd1 * dq(i, j, k, 1) + xi_x * (b3 - b4);
  convJacTimesDq[2] = v * b1 + xi_y * b2 + lmd1 * dq(i, j, k, 2) + xi_y * (b3 - b4);
  convJacTimesDq[3] = w * b1 + xi_z * b2 + lmd1 * dq(i, j, k, 3) + xi_z * (b3 - b4);
  convJacTimesDq[4] = h * b1 + U * b2 + lmd1 * dq(i, j, k, 4) + U * (b3 - b4);

  for (int l = 0; l < zone->n_scal; ++l) {
    convJacTimesDq[5 + l] = lmd1 * dq(i, j, k, l + 5) + sv(i, j, k, l) * b1;
  }
}

template<MixtureModel mixture_model, TurbMethod turb_method>
__global__ void DPLUR_inner_iteration(const DParameter *param, DZone *zone) {
  // This can be split into 3 kernels, such that the shared memory can be used.
  // E.g., i=2 needs ii=1 and ii=3, while i=4 needs ii=3 and ii=5, thus the ii=3 is recomputed.
  // If we use a kernel in i direction, with each thread computing an ii, for ii=-1~blockdim,
  // then all threads in the block except threadID=0 and blockDim, can use the just computed convJacTimesDq.
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const integer i = blockDim.x * blockIdx.x + threadIdx.x;
  const integer j = blockDim.y * blockIdx.y + threadIdx.y;
  const integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  const real dt_local = zone->dt_local(i, j, k);
  constexpr integer n_var_max = 5 + MAX_SPEC_NUMBER + 2; // 5+n_spec+n_turb(n_turb<=2)
  real convJacTimesDq[n_var_max], dq_total[n_var_max];

  const integer n_var{zone->n_var};
  const auto &inviscid_spectral_radius = zone->inv_spectr_rad;
  integer ii{i - 1}, jj{j - 1}, kk{k - 1};
  compute_jacobian_times_dq<mixture_model>(param, zone, ii, j, k, 0, inviscid_spectral_radius(ii, j, k)[0],
                                           convJacTimesDq);
  for (integer l = 0; l < n_var; ++l) {
    dq_total[l] = 0.5 * convJacTimesDq[l];
  }
  compute_jacobian_times_dq<mixture_model>(param, zone, i, jj, k, 1, inviscid_spectral_radius(i, jj, k)[1],
                                           convJacTimesDq);
  for (integer l = 0; l < n_var; ++l) {
    dq_total[l] += 0.5 * convJacTimesDq[l];
  }
  compute_jacobian_times_dq<mixture_model>(param, zone, i, j, kk, 2, inviscid_spectral_radius(i, j, kk)[2],
                                           convJacTimesDq);
  for (integer l = 0; l < n_var; ++l) {
    dq_total[l] += 0.5 * convJacTimesDq[l];
  }

  ii = i + 1;
  jj = j + 1;
  kk = k + 1;
  compute_jacobian_times_dq<mixture_model>(param, zone, ii, j, k, 0, -inviscid_spectral_radius(ii, j, k)[0],
                                           convJacTimesDq);
  for (integer l = 0; l < n_var; ++l) {
    dq_total[l] -= 0.5 * convJacTimesDq[l];
  }
  compute_jacobian_times_dq<mixture_model>(param, zone, i, jj, k, 1, -inviscid_spectral_radius(i, jj, k)[1],
                                           convJacTimesDq);
  for (integer l = 0; l < n_var; ++l) {
    dq_total[l] -= 0.5 * convJacTimesDq[l];
  }
  compute_jacobian_times_dq<mixture_model>(param, zone, i, j, kk, 2, -inviscid_spectral_radius(i, j, kk)[2],
                                           convJacTimesDq);
  for (integer l = 0; l < n_var; ++l) {
    dq_total[l] -= 0.5 * convJacTimesDq[l];
  }

  const auto &spect_rad = inviscid_spectral_radius(i, j, k);
  const real diag = 1 + dt_local * (spect_rad[0] + spect_rad[1] + spect_rad[2]);
  auto &dqk = zone->dqk;
  const auto &dq0 = zone->dq0;
  const integer n_spec{zone->n_spec};
  if constexpr (mixture_model == MixtureModel::Air || mixture_model == MixtureModel::Mixture) {
    for (integer l = 0; l < 5 + n_spec; ++l) {
      dqk(i, j, k, l) = dq0(i, j, k, l) + dt_local * dq_total[l] / diag;
    }
  } else if (mixture_model == MixtureModel::FR) {
    // Use point implicit method to dispose chemical source
    for (integer l = 0; l < 5; ++l) {
      dqk(i, j, k, l) = dq0(i, j, k, l) + dt_local * dq_total[l] / diag;
    }
    // Point implicit
  } // Flamelet method can be added later

  if constexpr (turb_method == TurbMethod::RANS) {
    // switch RANS model, and apply point implicit to treat the turbulent part
  }
}

template<MixtureModel mixture_model, TurbMethod turb_method>
void DPLUR(const Block &block, const DParameter *param, DZone *d_ptr, DZone *h_ptr, const Parameter &parameter) {
  const integer extent[3]{block.mx, block.my, block.mz};
  const integer dim{extent[2] == 1 ? 2 : 3};
  dim3 tpb{8, 8, 4};
  if (dim == 2) {
    tpb = {16, 16, 1};
  }
  const dim3 bpg{(extent[0] - 1) / tpb.x + 1, (extent[1] - 1) / tpb.y + 1, (extent[2] - 1) / tpb.z + 1};

  // DQ(0)=DQ/(1+dt*DRho+dt*dS/dQ)
  compute_DQ_0<mixture_model, turb_method><<<bpg, tpb>>>(d_ptr);
  // Take care of all such treatments where n_var is used to decide the memory size,
  // for when flamelet model is used, the data structure should be modified to make the useful data contigous.
  const auto mem_sz = h_ptr->dq.size() * h_ptr->n_var * sizeof(real);
  cudaMemcpy(h_ptr->dq0.data(), h_ptr->dq.data(), mem_sz, cudaMemcpyDeviceToDevice);

  for (integer iter = 0; iter < parameter.get_int("DPLUR_inner_step"); ++iter) {
    DPLUR_inner_iteration<mixture_model, turb_method><<<bpg, tpb>>>(param, d_ptr);
    // Theoretically, there should be a data communication here to exchange dq among processes.
    cudaMemcpy(h_ptr->dq.data(), h_ptr->dqk.data(), mem_sz, cudaMemcpyDeviceToDevice);
  }
}

}