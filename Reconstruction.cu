#include "Reconstruction.cuh"
#include "DParameter.h"
#include "Limiter.cuh"
#include "Constants.h"
#include "Thermo.cuh"

namespace cfd {
__host__ __device__ Reconstruction::Reconstruction(DParameter *param) : limiter{param->limiter} {

}

__device__ void
Reconstruction::apply(real *pv, real *pv_l, real *pv_r, const integer idx_shared, const integer n_var) {
  // The variables that can be reconstructed directly are density, u, v, w, p, Y_k, the number of which is
  // equal to the number of conservative variables(n_var).
  for (auto l = 0; l < n_var; +l) {
    pv_l[l] = pv[idx_shared * n_var + l];
    pv_r[l] = pv[(idx_shared + 1) * n_var + l];
  }
}

__host__ __device__ MUSCL::MUSCL(DParameter *param) : Reconstruction(param) {

}

__device__ void
MUSCL::apply(real *pv, real *pv_l, real *pv_r, const integer idx_shared, const integer n_var) {
  // The variables that can be reconstructed directly are density, u, v, w, p, Y_k, the number of which is
  // equal to the number of conservative variables(n_var).
  for (int l = 0; l < n_var; ++l) {
    // \Delta_i = u_i - u_{i-1}; \Delta_{i+1} = u_{i+1} - u_i
    const real delta_i{pv[idx_shared * n_var + l] - pv[(idx_shared - 1) * n_var + l]};
    const real delta_i1{pv[(idx_shared + 1) * n_var + l] - pv[idx_shared * n_var + l]};
    const real delta_i2{pv[(idx_shared + 2) * n_var + l] - pv[(idx_shared + 1) * n_var + l]};

    const real delta_neg_l = apply_limiter<0, 1>(limiter, delta_i, delta_i1);
    const real delta_pos_l = apply_limiter<0, 1>(limiter, delta_i1, delta_i);
    const real delta_neg_r = apply_limiter<0, 1>(limiter, delta_i1, delta_i2);
    const real delta_pos_r = apply_limiter<0, 1>(limiter, delta_i2, delta_i1);

    pv_l[l] = pv[idx_shared * n_var + l] + 0.25 * ((1 - kappa) * delta_neg_l + (1 + kappa) * delta_pos_l);
    pv_r[l] = pv[(idx_shared + 1) * n_var + l] - 0.25 * ((1 - kappa) * delta_pos_r + (1 + kappa) * delta_neg_r);
  }
}

__device__ void
reconstruction(real *pv, real *pv_l, real *pv_r, Reconstruction *&method, const integer idx_shared, DZone *zone,
               DParameter *param) {
  const auto n_var = zone->n_var;
  const auto n_spec = zone->n_spec;
  method->apply(pv, pv_l, pv_r, idx_shared, n_var);
  real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
  real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
#if MULTISPECIES == 1
  real mw_inv_l{0.0}, mw_inv_r{0.0};
  for (int l = 0; l < n_spec; ++l) {
    mw_inv_l += pv_l[5 + l] / param->mw[l];
    mw_inv_r += pv_r[5 + l] / param->mw[l];
  }
  const real t_l = pv_l[4] / (pv_l[0] * R_u * mw_inv_l);
  const real t_r = pv_r[4] / (pv_r[0] * R_u * mw_inv_r);

  real cv_l{0}, cv_r{0.0};
  real *mm = new real[4 * n_spec];
  real *hl = mm;
  real *hr = &mm[n_spec];
  real *cpl_i = &hr[n_spec];
  real *cpr_i = &cpl_i[n_spec];
  compute_enthalpy_and_cp(t_l, hl, cpl_i, param);
  compute_enthalpy_and_cp(t_r, hr, cpr_i, param);
  real cpl{0}, cpr{0}, cvl{0}, cvr{0};
  for (auto l = 0; l < n_spec; ++l) {
    cpl += cpl_i[l] * pv_l[l + 5];
    cpr += cpr_i[l] * pv_r[l + 5];
    cvl += pv_l[l + 5] * (cpl_i[l] - R_u / param->mw[l]);
    cvr += pv_r[l + 5] * (cpr_i[l] - R_u / param->mw[l]);
    el += hl[l] * pv_l[l + 5];
    er += hr[l] * pv_r[l + 5];
  }
  pv_l[5 + n_spec] = pv_l[0] * el - pv_l[4]; //total energy
  pv_r[5 + n_spec] = pv_r[0] * er - pv_r[4];

  pv_l[6 + n_spec] = cpl / cvl; //specific heat ratio
  pv_r[6 + n_spec] = cpr / cvr;
  delete[]mm;
#else
  pv_l[5]=el*pv_l[0]+pv_l[4] / (gamma_air - 1);
  pv_r[5]=er*pv_r[0]+pv_r[4] / (gamma_air - 1);
#endif
}
} // cfd