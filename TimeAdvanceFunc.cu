#include "TimeAdvanceFunc.cuh"
#include "Field.h"
#include "Thermo.cuh"

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

__global__ void cfd::limit_flow(cfd::DZone *zone, cfd::DParameter *param, integer blk_id) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const integer i = blockDim.x * blockIdx.x + threadIdx.x;
  const integer j = blockDim.y * blockIdx.y + threadIdx.y;
  const integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  // Record the computed values
  constexpr integer max_n_var = 5 + 2; // We don't limit the species mass fractions for now
  real var[max_n_var];
  var[0] = bv(i, j, k, 0);
  var[1] = bv(i, j, k, 1);
  var[2] = bv(i, j, k, 2);
  var[3] = bv(i, j, k, 3);
  var[4] = bv(i, j, k, 4);
  const integer n_spec{zone->n_spec};
  const integer n_turb = zone->n_scal - n_spec;
  for (integer l = 0; l < n_turb; ++l) {
    var[5 + l] = sv(i, j, k, l + n_spec);
  }

  // Find the unphysical values and limit them
  auto ll = param->limit_flow.ll;
  auto ul = param->limit_flow.ul;
  bool unphysical{false};
  const integer n_var = zone->n_var;
  for (integer l = 0; l < n_var; ++l) {
    if (isnan(var[l])) {
      unphysical = true;
      break;
    }
    if (var[l] < ll[l] || var[l] > ul[l]) {
      unphysical = true;
      break;
    }
  }

  if (unphysical) {
    printf("Unphysical values appear in process %d, block %d, i = %d, j = %d, k = %d.\n", param->myid, blk_id, i, j, k);

    real updated_var[max_n_var + MAX_SPEC_NUMBER];
    memset(updated_var, 0, max_n_var * sizeof(real));
    integer kn{0};
    // Compute the sum of all "good" points surrounding the "bad" point
    for (integer ka = -1; ka < 2; ++ka) {
      const integer k1{k + ka};
      if (k1 < 0 || k1 >= mz) return;
      for (integer ja = -1; ja < 2; ++ja) {
        const integer j1{j + ja};
        if (j1 < 0 || j1 >= my) return;
        for (integer ia = -1; ia < 2; ++ia) {
          const integer i1{i + ia};
          if (i1 < 0 || i1 >= mz)return;

          unphysical = false;
          for (integer l = 0; l < 5; ++l) {
            const auto value{bv(i1, j1, k1, l)};
            if (isnan(value) || value < ll[l] || value > ul[l]) {
              unphysical = true;
              break;
            }
            updated_var[l] += value;
          }
          if (unphysical) continue;

          for (integer l = 0; l < n_turb; ++l) {
            const auto value{sv(i1, j1, k1, l + n_spec)};
            if (isnan(value) || value < ll[l + 5] || value > ul[l + 5]) {
              unphysical = true;
              break;
            }
            updated_var[l + 5 + n_spec] += value;
          }
          if (unphysical) continue;

          for (integer l = 0; l < n_spec; ++l) {
            updated_var[l + 5] += sv(i1, j1, k1, l);
          }

          ++kn;
        }
      }
    }

    // Compute the average of the surrounding points
    if (kn > 0) {
      for (integer l = 0; l < n_var; ++l) {
        updated_var[l] /= kn;
      }
    } else {
      // The surrounding points are all "bad"
      for (integer l = 0; l < 5; ++l) {
        updated_var[l] = max(var[l], ll[l]);
        updated_var[l] = min(updated_var[l], ul[l]);
      }
      for (integer l = 0; l < n_spec; ++l) {
        updated_var[l + 5] = param->limit_flow.sv_inf[l];
      }
      if (param->rans_model == 2) {
        updated_var[5 + n_spec] = var[5 + n_spec];
        updated_var[6 + n_spec] = var[6 + n_spec];

        if (updated_var[5 + n_spec] < 0) {
          updated_var[5 + n_spec] = param->limit_flow.sv_inf[n_spec];
        }
        if (updated_var[6 + n_spec] < 0) {
          updated_var[6 + n_spec] = param->limit_flow.sv_inf[n_spec + 1];
        }
      }
    }

    // Assign averaged values for the bad point
    auto &cv = zone->cv;
    bv(i, j, k, 0) = updated_var[0];
    bv(i, j, k, 1) = updated_var[1];
    bv(i, j, k, 2) = updated_var[2];
    bv(i, j, k, 3) = updated_var[3];
    bv(i, j, k, 4) = updated_var[4];
    cv(i, j, k, 0) = updated_var[0];
    cv(i, j, k, 1) = updated_var[0] * updated_var[1];
    cv(i, j, k, 2) = updated_var[0] * updated_var[2];
    cv(i, j, k, 3) = updated_var[0] * updated_var[3];
    cv(i, j, k, 4) = 0.5 * updated_var[0] * (updated_var[1] * updated_var[1] + updated_var[2] * updated_var[2] +
                                             updated_var[3] * updated_var[3]);
    const integer n_scalar{param->n_scalar};
    for (integer l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = updated_var[5 + l];
      cv(i, j, k, 5 + l) = updated_var[0] * updated_var[5 + l];
    }
    if (n_spec > 0) {
      real mw = 0;
      for (integer l = 0; l < n_spec; ++l) {
        mw += sv(i, j, k, l) / param->mw[l];
      }
      mw = 1 / mw;
      bv(i, j, k, 5) = updated_var[4] * mw / (updated_var[0] * R_u);
      real enthalpy[MAX_SPEC_NUMBER];
      compute_enthalpy(bv(i, j, k, 5), enthalpy, param);
      // Add species enthalpy together up to kinetic energy to get total enthalpy
      for (auto l = 0; l < zone->n_spec; l++) {
        cv(i, j, k, 4) += enthalpy[l] * cv(i, j, k, 5 + l);
      }
      cv(i, j, k, 4) -= bv(i, j, k, 4);  // (\rho e =\rho h - p)
    } else {
      bv(i, j, k, 5) = updated_var[4] * mw_air / (updated_var[0] * R_u);
      cv(i, j, k, 4) += updated_var[4] / (gamma_air - 1);
    }
  }
}

