#include "Thermo.cuh"
#include "DParameter.h"
#include "Constants.h"
#include "Field.h"

#if MULTISPECIES==1
__device__ void cfd::compute_enthalpy(real t, real *enthalpy, cfd::DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  for (int i = 0; i < param->n_spec; ++i) {
    if (t < param->t_low[i]) {
      const real tt = param->t_low[i];
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
      auto &coeff = param->low_temp_coeff;
      enthalpy[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
                    0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
      const real cp = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
      enthalpy[i] += cp * (t - tt); // Do a linear interpolation for enthalpy
    } else {
      auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
      enthalpy[i] = coeff(i, 0) * t + 0.5 * coeff(i, 1) * t2 + coeff(i, 2) * t3 / 3 + 0.25 * coeff(i, 3) * t4 +
                    0.2 * coeff(i, 4) * t5 + coeff(i, 5);
    }
    enthalpy[i] *= cfd::R_u / param->mw[i];
  }
}

__device__ void cfd::compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, const DParameter *param) {
  const double t2{t * t}, t3{t2 * t}, t4{t3 * t}, t5{t4 * t};
  for (int i = 0; i < param->n_spec; ++i) {
    if (t < param->t_low[i]) {
      const double tt = param->t_low[i];
      const double tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt, tt5 = tt4 * tt;
      auto &coeff = param->low_temp_coeff;
      enthalpy[i] = coeff(i, 0) * tt + 0.5 * coeff(i, 1) * tt2 + coeff(i, 2) * tt3 / 3 + 0.25 * coeff(i, 3) * tt4 +
                    0.2 * coeff(i, 4) * tt5 + coeff(i, 5);
      cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
      enthalpy[i] += cp[i] * (t - tt); // Do a linear interpolation for enthalpy
    } else {
      auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
      enthalpy[i] = coeff(i, 0) * t + 0.5 * coeff(i, 1) * t2 + coeff(i, 2) * t3 / 3 + 0.25 * coeff(i, 3) * t4 +
                    0.2 * coeff(i, 4) * t5 + coeff(i, 5);
      cp[i] = coeff(i, 0) + coeff(i, 1) * t + coeff(i, 2) * t2 + coeff(i, 3) * t3 + coeff(i, 4) * t4;
    }
    enthalpy[i] *= R_u / param->mw[i];
    cp[i] *= R_u / param->mw[i];
  }
}

__device__ void cfd::compute_cp(real t, real *cp, cfd::DParameter *param) {
  const real t2{t * t}, t3{t2 * t}, t4{t3 * t};
  for (auto i = 0; i < param->n_spec; ++i) {
    if (t < param->t_low[i]) {
      const real tt = param->t_low[i];
      const real tt2 = tt * tt, tt3 = tt2 * tt, tt4 = tt3 * tt;
      auto &coeff = param->low_temp_coeff;
      cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 + coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
    } else {
      auto &coeff = t < param->t_mid[i] ? param->low_temp_coeff : param->high_temp_coeff;
      cp[i] = coeff(i, 0) + coeff(i, 1) * t + coeff(i, 2) * t2 + coeff(i, 3) * t3 + coeff(i, 4) * t4;
    }
    cp[i] *= R_u / param->mw[i];
  }
}
#endif

__device__ void cfd::compute_total_energy(integer i, integer j, integer k, cfd::DZone *zone, DParameter *param) {
  auto &bv = zone->bv;
  auto &vel = zone->vel;
  auto &cv = zone->cv;

  vel(i, j, k) = bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3);
  cv(i, j, k, 4) = 0.5 * bv(i, j, k, 0) * vel(i, j, k);
#if MULTISPECIES == 1
//  real *enthalpy = new real[zone->n_spec];
  real enthalpy[9];
  compute_enthalpy(bv(i, j, k, 5), enthalpy, param);
  // Add species enthalpy together up to kinetic energy to get total enthalpy
  for (auto l = 0; l < zone->n_spec; l++) {
    cv(i, j, k, 4) += enthalpy[l] * cv(i, j, k, 5 + l);
  }
  cv(i, j, k, 4) -= bv(i, j, k, 4);  // (\rho e =\rho h - p)
//  delete[]enthalpy;
#else
  cv(i, j, k, 4) += bv(i, j, k, 4) / (cfd::gamma_air - 1);
#endif // MULTISPECIES==1
  vel(i, j, k) = sqrt(vel(i, j, k));
}

#if MULTISPECIES==1
__device__ void cfd::compute_temperature(int i, int j, int k, const cfd::DParameter *param, cfd::DZone *zone) {
  const integer n_spec=param->n_spec;
  auto& Y=zone->yk;
  auto& Q=zone->cv;
  auto& bv=zone->bv;

  real mw{0};
  for (integer l = 0; l < n_spec; ++l)
    mw += Y(i, j, k, l) / param->mw[l];
  mw = 1 / mw;
  const real gas_const = R_u / mw;
  const real e = Q(i, j, k, 4) / Q(i, j, k, 0) - 0.5 * (bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) + bv(i, j, k, 3) * bv(i, j, k, 3));

  real err{1}, t{bv(i, j, k, 5)};
  constexpr integer max_iter{1000};
  constexpr real eps{1e-3};
  integer iter = 0;

  constexpr integer nsp=60;
  real h_i[nsp],cp_i[nsp];
  while (err > eps && iter++ < max_iter) {
    compute_enthalpy_and_cp(t,h_i,cp_i,param);
    real cp_tot{0}, h{0};
    for (integer l=0;l<n_spec;++l){
      cp_tot+=cp_i[l]*Y(i,j,k,l);
      h+=h_i[l]*Y(i,j,k,l);
    }
    const real e_t    = h - gas_const * t;
    const real cv     = cp_tot - gas_const;
    const real t1     = t - (e_t - e) / cv;
    err = std::abs(1 - t1 / t);
    t = t1;
  }
  bv(i, j, k, 5) = t;
  bv(i, j, k, 4) = bv(i, j, k, 0) * t * gas_const;
}
#endif