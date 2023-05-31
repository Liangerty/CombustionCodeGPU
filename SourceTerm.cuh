#pragma once

#include "Define.h"
#include "Mesh.h"
#include "Field.h"
#include "DParameter.h"

namespace cfd {
template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void compute_source(cfd::DZone *zone, DParameter *param) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  auto &dq = zone->dq;

  if constexpr (turb_method == TurbMethod::RANS) {
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
        const real u_x = 0.5 * (xi_x * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                                eta_x * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                                zeta_x * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
        const real u_y = 0.5 * (xi_y * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                                eta_y * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                                zeta_y * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
        const real u_z = 0.5 * (xi_z * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                                eta_z * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                                zeta_z * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
        const real v_x = 0.5 * (xi_x * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                                eta_x * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                                zeta_x * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
        const real v_y = 0.5 * (xi_y * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                                eta_y * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                                zeta_y * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
        const real v_z = 0.5 * (xi_z * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                                eta_z * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                                zeta_z * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
        const real w_x = 0.5 * (xi_x * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                                eta_x * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                                zeta_x * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
        const real w_y = 0.5 * (xi_y * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                                eta_y * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                                zeta_y * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
        const real w_z = 0.5 * (xi_z * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                                eta_z * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                                zeta_z * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));

        // First, compute the turbulent viscosity.
        // Theoretically, this should be computed after updating the basic variables, but after that we won't need it until now.
        // Besides, we need the velocity gradients in the computation, which are also needed when computing source terms.
        // In order to alleviate the computational burden, we put the computation of mut here.
        const integer n_spec{zone->n_spec};
        const real rhoK = zone->cv(i, j, k, n_spec + 5);
        const real tke = zone->sv(i, j, k, n_spec);
        const real omega = zone->sv(i, j, k, n_spec + 1);
        const real vorticity = (v_x - u_y) * (v_x - u_y) + (w_x - u_z) * (w_x - u_z) + (w_y - v_z) * (w_y - v_z);
        const real density = zone->bv(i, j, k, 0);

        // If wall, mut=0. Else, compute mut as in the if statement.
        real f2{1};
        const real dy = zone->wall_distance(i, j, k);
        if (dy > 1e-25) {
          const real param1 = 2 * std::sqrt(tke) / (0.09 * omega * dy);
          const real param2 = 500 * zone->mul(i, j, k) / (density * dy * dy);
          const real arg2 = max(param1, param2);
          f2 = std::tanh(arg2 * arg2);
        }
        real mut{0};
        if (const real denominator = max(SST::a_1 * omega, vorticity * f2); denominator > 1e-25) {
          mut = SST::a_1 * rhoK / denominator;
        }
        zone->mut(i, j, k) = mut;

        // Next, compute the source term for turbulent kinetic energy.
        const real divU = u_x + v_y + w_z;
        const real divU2 = divU * divU;

        const real prod_k = mut * (2 * (u_x * u_x + v_y * v_y + w_z * w_z) - 2 / 3 * divU2 + (u_y + v_x) * (u_y + v_x) +
                                   (u_z + w_x) * (u_z + w_x) + (v_z + w_y) * (v_z + w_y)) - 2 / 3 * rhoK * divU;
        const real diss_k = SST::beta_star * rhoK * omega;
        const real jac = zone->jac(i, j, k);
        dq(i, j, k, n_spec + 5) += jac * (prod_k - diss_k);

        // omega source term
        auto& sv=zone->sv;
        const real k_x = 0.5 * (xi_x * (sv(i + 1, j, k, n_spec) - sv(i - 1, j, k, n_spec)) +
                                eta_x * (sv(i, j + 1, k, n_spec) - sv(i, j - 1, k, n_spec)) +
                                zeta_x * (sv(i, j, k + 1, n_spec) - sv(i, j, k - 1, n_spec)));
        const real k_y = 0.5 * (xi_y * (sv(i + 1, j, k, n_spec) - sv(i - 1, j, k, n_spec)) +
                                eta_y * (sv(i, j + 1, k, n_spec) - sv(i, j - 1, k, n_spec)) +
                                zeta_y * (sv(i, j, k + 1, n_spec) - sv(i, j, k - 1, n_spec)));
        const real k_z = 0.5 * (xi_z * (sv(i + 1, j, k, n_spec) - sv(i - 1, j, k, n_spec)) +
                                eta_z * (sv(i, j + 1, k, n_spec) - sv(i, j - 1, k, n_spec)) +
                                zeta_z * (sv(i, j, k + 1, n_spec) - sv(i, j, k - 1, n_spec)));

        const real omega_x = 0.5 * (xi_x * (sv(i + 1, j, k, n_spec+1) - sv(i - 1, j, k, n_spec+1)) +
                                    eta_x * (sv(i, j + 1, k, n_spec+1) - sv(i, j - 1, k, n_spec+1)) +
                                    zeta_x * (sv(i, j, k + 1, n_spec+1) - sv(i, j, k - 1, n_spec+1)));
        const real omega_y = 0.5 * (xi_y * (sv(i + 1, j, k, n_spec+1) - sv(i - 1, j, k, n_spec+1)) +
                                    eta_y * (sv(i, j + 1, k, n_spec+1) - sv(i, j - 1, k, n_spec+1)) +
                                    zeta_y * (sv(i, j, k + 1, n_spec+1) - sv(i, j, k - 1, n_spec+1)));
        const real omega_z =0.5 * (xi_z * (sv(i + 1, j, k, n_spec+1) - sv(i - 1, j, k, n_spec+1)) +
                                   eta_z * (sv(i, j + 1, k, n_spec+1) - sv(i, j - 1, k, n_spec+1)) +
                                   zeta_z * (sv(i, j, k + 1, n_spec+1) - sv(i, j, k - 1, n_spec+1)));
        const real inter_var=2 * density * cfd::SST::sigma_omega2 / omega * (k_x * omega_x + k_y * omega_y + k_z * omega_z);

        real f1{1};
        if (dy > 1e-25) {
          const real param1{std::sqrt(tke) / (0.09 * omega * dy)};

          const real d2 = dy * dy;
          const real param2{500 * zone->mul(i, j, k) / (density * d2 * omega)};
          const real CDkomega{max(1e-20, inter_var)};
          const real param3{4 * density * SST::sigma_omega2 * tke / (CDkomega * d2)};

          const real arg1{min(max(param1, param2), param3)};
          f1 = std::tanh(arg1 * arg1 * arg1 * arg1);
        }

        const real gamma=SST::gamma2+SST::delta_gamma*f1;
        const real prod_omega = gamma * density / mut * prod_k + (1 - f1)*inter_var;
        const real beta=SST::beta_2+SST::delta_beta*f1;
        const real diss_omega=-beta*density*omega*omega;
        dq(i,j,k,n_spec+6)+=jac*(prod_omega-diss_omega);
    }
  }
}
}