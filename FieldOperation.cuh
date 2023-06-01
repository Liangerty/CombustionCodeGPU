// Contains some operations on different variables.
// E.g., total energy is computed from V and h; conservative variables are computed from basic variables
#pragma once

#include "Define.h"
#include "Field.h"
#include "Thermo.cuh"
#include "Constants.h"
#include "Transport.cuh"
#include "DParameter.h"

namespace cfd {

template<MixtureModel mixture_model>
__device__ void compute_total_energy(integer i, integer j, integer k, cfd::DZone *zone, DParameter *param) {
  auto &bv = zone->bv;
  auto &vel = zone->vel;
  auto &cv = zone->cv;

  vel(i, j, k) = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3);
  cv(i, j, k, 4) = 0.5 * bv(i, j, k, 0) * vel(i, j, k);
  if constexpr (mixture_model != MixtureModel::Air) {
    real enthalpy[MAX_SPEC_NUMBER];
    compute_enthalpy(bv(i, j, k, 5), enthalpy, param);
    // Add species enthalpy together up to kinetic energy to get total enthalpy
    for (auto l = 0; l < zone->n_spec; l++) {
      cv(i, j, k, 4) += enthalpy[l] * cv(i, j, k, 5 + l);
    }
    cv(i, j, k, 4) -= bv(i, j, k, 4);  // (\rho e =\rho h - p)
  } else {
    cv(i, j, k, 4) += bv(i, j, k, 4) / (cfd::gamma_air - 1);
  }
  vel(i, j, k) = sqrt(vel(i, j, k));
}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void compute_cv_from_bv(DZone *zone, DParameter *param) {
  const integer ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const auto &bv = zone->bv;
  auto &cv = zone->cv;
  const real rho = bv(i, j, k, 0);
  const real u = bv(i, j, k, 1);
  const real v = bv(i, j, k, 2);
  const real w = bv(i, j, k, 3);

  cv(i, j, k, 0) = rho;
  cv(i, j, k, 1) = rho * u;
  cv(i, j, k, 2) = rho * v;
  cv(i, j, k, 3) = rho * w;
  // It seems we don't need an if here, if there is no other scalars, n_scalar=0; else, n_scalar=n_spec+n_turb
  const integer n_scalar{zone->n_scal};
  const auto &sv = zone->sv;
  if constexpr (mix_model != MixtureModel::FL) {
    for (auto l = 0; l < n_scalar; ++l) {
      cv(i, j, k, 5 + l) = rho * sv(i, j, k, l);
    }
  } else {
    // Flamelet model
    const integer n_spec{zone->n_spec};
    integer n_eqn{n_scalar - n_spec};
    for (auto l = 0; l < n_eqn; ++l) {
      cv(i, j, k, 5 + l) = rho * sv(i, j, k, l + n_spec);
    }
  }

  compute_total_energy<mix_model>(i, j, k, zone, param);
}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void update_physical_properties(DZone *zone, DParameter *param) {
  const integer ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const real temperature{zone->bv(i, j, k, 5)};
  if constexpr (mix_model != MixtureModel::Air) {
    const integer n_spec{zone->n_spec};
    auto &yk = zone->sv;
    real mw{0}, cp_tot{0}, cv{0};
    real cp[MAX_SPEC_NUMBER];
    compute_cp(temperature, cp, param);
    for (auto l = 0; l < n_spec; ++l) {
      mw += yk(i, j, k, l) / param->mw[l];
      cp_tot += yk(i, j, k, l) * cp[l];
      cv += yk(i, j, k, l) * (cp[l] - R_u / param->mw[l]);
    }
    mw = 1 / mw;
    zone->cp(i, j, k) = cp_tot;
    zone->gamma(i, j, k) = cp_tot / cv;
    zone->acoustic_speed(i, j, k) = std::sqrt(zone->gamma(i, j, k) * R_u * temperature / mw);
    compute_transport_property(i, j, k, temperature, mw, cp, param, zone);
  } else {
    constexpr real c_temp{gamma_air * R_u / mw_air};
    constexpr real cp{gamma_air * R_u / mw_air / (gamma_air - 1)};
    const real pr = param->Pr;
    zone->acoustic_speed(i, j, k) = std::sqrt(c_temp * temperature);
    zone->mul(i, j, k) = Sutherland(temperature);
    zone->thermal_conductivity(i, j, k) = zone->mul(i, j, k) * cp / pr;
  }
  zone->mach(i, j, k) = zone->vel(i, j, k) / zone->acoustic_speed(i, j, k);
}

template<MixtureModel mix_model>
__global__ void initialize_mut(DZone *zone, DParameter *param) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - 1;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - 1;
  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - 1;
  if (i >= mx + 1 || j >= my + 1 || k >= mz + 1) return;

  auto &dq = zone->dq;

  switch (param->rans_model) {
    case 1://SA
      break;
    case 2:
    default: // SST
      const auto &m = zone->metric(i, j, k);
      const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
      const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
      const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};

      // Compute the gradient of velocity
      const auto &bv = zone->bv;
      const real u_y = 0.5 * (xi_y * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                              eta_y * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                              zeta_y * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
      const real u_z = 0.5 * (xi_z * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                              eta_z * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                              zeta_z * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
      const real v_x = 0.5 * (xi_x * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                              eta_x * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                              zeta_x * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
      const real v_z = 0.5 * (xi_z * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                              eta_z * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                              zeta_z * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
      const real w_x = 0.5 * (xi_x * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                              eta_x * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                              zeta_x * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
      const real w_y = 0.5 * (xi_y * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                              eta_y * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                              zeta_y * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));

      // First, compute the turbulent viscosity.
      // Theoretically, this should be computed after updating the basic variables, but after that we won't need it until now.
      // Besides, we need the velocity gradients in the computation, which are also needed when computing source terms.
      // In order to alleviate the computational burden, we put the computation of mut here.
      const integer n_spec{zone->n_spec};
      const real rhoK = zone->cv(i, j, k, n_spec + 5);
      const real tke = zone->sv(i, j, k, n_spec);
      const real omega = zone->sv(i, j, k, n_spec + 1);
      const real vorticity = std::sqrt((v_x - u_y) * (v_x - u_y) + (w_x - u_z) * (w_x - u_z) + (w_y - v_z) * (w_y - v_z));
      const real density = zone->bv(i, j, k, 0);

      // If wall, mut=0. Else, compute mut as in the if statement.
      real f2{1};
      const real dy = zone->wall_distance(i, j, k);
      if (dy > 1e-25) {
        const real param1 = 2 * std::sqrt(tke) / (0.09 * omega * dy);
        const real temperature{zone->bv(i, j, k, 5)};
        real mul = Sutherland(temperature);
        if constexpr (mix_model != MixtureModel::Air) {
          auto &yk = zone->sv;
          real mw{0};
          for (auto l = 0; l < n_spec; ++l) {
            mw += yk(i, j, k, l) / param->mw[l];
          }
          mw = 1 / mw;
          mul = compute_viscosity(i, j, k, temperature, mw, param, zone);
        }
        const real param2 = 500 * mul / (density * dy * dy * omega);
        const real arg2 = max(param1, param2);
        f2 = std::tanh(arg2 * arg2);
      }
      real mut{0};
      if (const real denominator = max(SST::a_1 * omega, vorticity * f2); denominator > 1e-25) {
        mut = SST::a_1 * rhoK / denominator;
      }
      zone->mut(i, j, k) = mut;
  }
}

__device__ void compute_temperature(int i, int j, int k, const cfd::DParameter *param, cfd::DZone *zone);

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void update_cv_and_bv(cfd::DZone *zone, DParameter *param) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &cv = zone->cv;

  real dt_div_jac = zone->dt_local(i, j, k) / zone->jac(i, j, k);
  for (integer l = 0; l < zone->n_var; ++l) {
    cv(i, j, k, l) += zone->dq(i, j, k, l) * dt_div_jac;
  }
  if (extent[2] == 1) {
    cv(i, j, k, 3) = 0;
  }

  auto &bv = zone->bv;
  auto &velocity = zone->vel(i, j, k);

  bv(i, j, k, 0) = cv(i, j, k, 0);
  const real density_inv = 1.0 / cv(i, j, k, 0);
  bv(i, j, k, 1) = cv(i, j, k, 1) * density_inv;
  bv(i, j, k, 2) = cv(i, j, k, 2) * density_inv;
  bv(i, j, k, 3) = cv(i, j, k, 3) * density_inv;
  velocity = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3); //V^2

  auto &sv = zone->sv;
  if constexpr (mix_model != MixtureModel::Air ||
                turb_method == TurbMethod::RANS) { // Flamelet method should be written independently.
    // For multiple species or RANS methods, there will be scalars to be computed
    for (integer l = 0; l < zone->n_scal; ++l) {
      sv(i, j, k, l) = cv(i, j, k, 5 + l) * density_inv;
    }
  }
  if constexpr (mix_model != MixtureModel::Air) {
    compute_temperature(i, j, k, param, zone);
  } else {
    // Air
    bv(i, j, k, 4) = (gamma_air - 1) * (cv(i, j, k, 4) - 0.5 * bv(i, j, k, 0) * velocity);
    bv(i, j, k, 5) = bv(i, j, k, 4) * mw_air * density_inv / R_u;
  }
  velocity = std::sqrt(velocity);
}

__global__ void eliminate_k_gradient(cfd::DZone *zone);
}