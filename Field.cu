#include "Field.h"
#include "Mesh.h"
#include "BoundCond.h"
#include "Thermo.cuh"
#include "DParameter.h"
#include "Transport.cuh"

cfd::HZone::HZone(Parameter &parameter, const Block &block) {
  const integer mx{block.mx}, my{block.my}, mz{block.mz}, ngg{block.ngg};
  const integer n_var{parameter.get_int("n_var")};

  cv.resize(mx, my, mz, n_var, ngg);
  bv.resize(mx, my, mz, 6, ngg);
  mach.resize(mx, my, mz, ngg);
#if MULTISPECIES == 1
  const integer n_spec{parameter.get_int("n_spec")};
  yk.resize(mx, my, mz, n_spec, ngg);
#endif
#ifdef _DEBUG
  dbv_squared.resize(mx, my, mz, 4, 0);
  tempo_var.resize(mx, my, mz, 0);
  dq.resize(mx, my, mz, n_var, 0);
#endif
}

void cfd::HZone::initialize_basic_variables(const cfd::Parameter &parameter, const cfd::Block &block,
                                            const std::vector<Inflow> &inflows, const std::vector<real> &xs,
                                            const std::vector<real> &xe, const std::vector<real> &ys,
                                            const std::vector<real> &ye, const std::vector<real> &zs,
                                            const std::vector<real> &ze) {
  const auto n = inflows.size();
  std::vector<real> rho(n, 0), u(n, 0), v(n, 0), w(n, 0), p(n, 0), T(n, 0);
#if MULTISPECIES == 1
  const auto n_spec = parameter.get_int("n_spec");
  gxl::MatrixDyn<double> mass_frac{static_cast<int>(n), n_spec};
#endif // MULTISPECIES==1
  for (size_t i = 0; i < inflows.size(); ++i) {
    std::tie(rho[i], u[i], v[i], w[i], p[i], T[i]) = inflows[i].var_info();
#if MULTISPECIES == 1
    auto y_spec = inflows[i].yk;
    for (int k = 0; k < n_spec; ++k) {
      mass_frac(static_cast<int>(i), k) = y_spec[k];
    }
#endif
  }

  const int ngg{block.ngg};
  for (int i = -ngg; i < block.mx + ngg; ++i) {
    for (int j = -ngg; j < block.my + ngg; ++j) {
      for (int k = -ngg; k < block.mz + ngg; ++k) {
        size_t i_init{0};
        if (inflows.size() > 1) {
          for (size_t l = 1; l < inflows.size(); ++l) {
            if (block.x(i, j, k) >= xs[l] && block.x(i, j, k) <= xe[l]
                && block.y(i, j, k) >= ys[l] && block.y(i, j, k) <= ye[l]
                && block.z(i, j, k) >= zs[l] && block.z(i, j, k) <= ze[l]) {
              i_init = l;
              break;
            }
          }
        }
        bv(i, j, k, 0) = rho[i_init];
        bv(i, j, k, 1) = u[i_init];
        bv(i, j, k, 2) = v[i_init];
        bv(i, j, k, 3) = w[i_init];
        bv(i, j, k, 4) = p[i_init];
        bv(i, j, k, 5) = T[i_init];
#if MULTISPECIES == 1
        for (int l = 0; l < n_spec; ++l) {
          yk(i, j, k, l) = mass_frac(static_cast<int>(i_init), l);
        }
#endif // MULTISPECIES==1
      }
    }
  }
}

cfd::Field::Field(Parameter &parameter, const Block &block)
    : h_zone(parameter, block) {}

void cfd::Field::initialize_basic_variables(const cfd::Parameter &parameter, const cfd::Block &block,
                                            const std::vector<Inflow> &inflows, const std::vector<real> &xs,
                                            const std::vector<real> &xe, const std::vector<real> &ys,
                                            const std::vector<real> &ye, const std::vector<real> &zs,
                                            const std::vector<real> &ze) {
  h_zone.initialize_basic_variables(parameter, block, inflows, xs, xe, ys, ye, zs, ze);
}

void cfd::Field::setup_device_memory(const Parameter &parameter, const Block &b) {
  h_ptr = new DZone;
  h_ptr->mx = b.mx, h_ptr->my = b.my, h_ptr->mz = b.mz, h_ptr->ngg = b.ngg;

  h_ptr->x.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  cudaMemcpy(h_ptr->x.data(), b.x.data(), sizeof(real) * h_ptr->x.size(), cudaMemcpyHostToDevice);
  h_ptr->y.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  cudaMemcpy(h_ptr->y.data(), b.y.data(), sizeof(real) * h_ptr->y.size(), cudaMemcpyHostToDevice);
  h_ptr->z.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  cudaMemcpy(h_ptr->z.data(), b.z.data(), sizeof(real) * h_ptr->z.size(), cudaMemcpyHostToDevice);

  auto n_bound{b.boundary.size()};
  auto n_inner{b.inner_face.size()};
  auto n_par{b.parallel_face.size()};
  auto mem_sz = sizeof(Boundary) * n_bound;
  cudaMalloc(&h_ptr->boundary, mem_sz);
  cudaMemcpy(h_ptr->boundary, b.boundary.data(), mem_sz, cudaMemcpyHostToDevice);
  mem_sz = sizeof(InnerFace) * n_inner;
  cudaMalloc(&h_ptr->innerface, mem_sz);
  cudaMemcpy(h_ptr->innerface, b.inner_face.data(), mem_sz, cudaMemcpyHostToDevice);
  mem_sz = sizeof(ParallelFace) * n_par;
  cudaMalloc(&h_ptr->parface, mem_sz);
  cudaMemcpy(h_ptr->parface, b.parallel_face.data(), mem_sz, cudaMemcpyHostToDevice);

  h_ptr->jac.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  cudaMemcpy(h_ptr->jac.data(), b.jacobian.data(), sizeof(real) * h_ptr->jac.size(), cudaMemcpyHostToDevice);
  h_ptr->metric.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  cudaMemcpy(h_ptr->metric.data(), b.metric.data(), sizeof(gxl::Matrix<real, 3, 3, 1>) * h_ptr->metric.size(),
             cudaMemcpyHostToDevice);

  h_ptr->n_var = parameter.get_int("n_var");
  h_ptr->cv.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_var, h_ptr->ngg);
  h_ptr->bv.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 6, h_ptr->ngg);
  cudaMemcpy(h_ptr->bv.data(), h_zone.bv.data(), sizeof(real) * h_ptr->bv.sz * 6, cudaMemcpyHostToDevice);
  h_ptr->bv_last.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 4, 0);
  h_ptr->vel.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  h_ptr->acoustic_speed.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  h_ptr->mach.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  h_ptr->mul.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  h_ptr->conductivity.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
#if MULTISPECIES == 1
  h_ptr->n_spec = parameter.get_int("n_spec");
  h_ptr->yk.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_spec, h_ptr->ngg);
  cudaMemcpy(h_ptr->yk.data(), h_zone.yk.data(), sizeof(real) * h_ptr->yk.sz * h_ptr->n_spec, cudaMemcpyHostToDevice);
  h_ptr->rho_D.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_spec, h_ptr->ngg);
  h_ptr->gamma.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
#endif // MULTISPECIES==1
  h_ptr->dq.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_var, 0);
  if (parameter.get_int("temporal_scheme") == 1) {//LUSGS
    h_ptr->inv_spectr_rad.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 0);
    h_ptr->visc_spectr_rad.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 0);
  }
  if (parameter.get_bool("steady")) { // steady simulation
    h_ptr->dt_local.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 0);
  }

  cudaMalloc(&d_ptr, sizeof(DZone));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DZone), cudaMemcpyHostToDevice);
}

__global__ void cfd::compute_cv_from_bv(cfd::DZone *zone, cfd::DParameter *param) {
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
#if MULTISPECIES == 1
  const integer n_spec{zone->n_spec};
  const auto &yk = zone->yk;
  for (auto l = 0; l < n_spec; ++l) {
    cv(i, j, k, 5 + l) = rho * yk(i, j, k, l);
  }
#endif // MULTISPECIES==1
  compute_total_energy(i, j, k, zone, param);
}

__global__ void cfd::update_physical_properties(cfd::DZone *zone, cfd::DParameter *param) {
  const integer ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const real temperature{zone->bv(i, j, k, 5)};
#if MULTISPECIES == 1
  const integer n_spec{zone->n_spec};
  auto &yk = zone->yk;
  real mw{0}, cp_tot{0}, cv{0};
  real *cp = new real[n_spec];
  compute_cp(temperature, cp, param);
  for (auto l = 0; l < n_spec; ++l) {
    mw += yk(i, j, k, l) / param->mw[l];
    cp_tot += yk(i, j, k, l) * cp[l];
    cv += yk(i, j, k, l) * (cp[l] - R_u / param->mw[l]);
  }
  mw = 1 / mw;
  zone->gamma(i, j, k) = cp_tot / cv;
  zone->acoustic_speed(i, j, k) = std::sqrt(zone->gamma(i, j, k) * R_u * temperature / mw);
  compute_transport_property(i, j, k, temperature, mw, cp, param, zone);
  delete[] cp;
#else
  constexpr real c_temp{gamma_air * R_u / mw_air};
  const real pr = param->Pr;
  zone->acoustic_speed(i,j,k) = std::sqrt(c_temp * temperature);
  zone->mul(i, j, k) = Sutherland(temperature);
  zone->conductivity(i, j, k) = zone->mul(i, j, k) * c_temp / (gamma_air - 1) / pr;
#endif
  zone->mach(i, j, k) = zone->vel(i, j, k) / zone->acoustic_speed(i, j, k);
}


__global__ void cfd::inner_communication(cfd::DZone *zone, cfd::DZone *tar_zone, const uint *n_point, integer i_face) {
  uint n[3];
  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
  n[1] = blockDim.y + blockIdx.y + threadIdx.y;
  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
  if (n[0] >= n_point[0] || n[1] >= n_point[1] || n[2] >= n_point[2]) return;

  integer idx[3], idx_tar[3];
  const auto &f = zone->innerface[i_face];
  for (integer i = 0; i < 3; ++i) {
    auto d_idx = f.loop_dir[i] * (integer) (n[i]);
    idx[i] = f.range_start[i] + d_idx;
    idx_tar[i] = f.target_start[i] + f.target_loop_dir[i] * d_idx;
  }

  // The face direction: which of i(0)/j(1)/k(2) is the coincided face.
  const auto face_dir{f.direction > 0 ? f.range_start[f.face] : f.range_end[f.face]};

  if (idx[f.face] == face_dir) {
    // If this is the corresponding face, then average the values from both blocks
    for (integer l = 0; l < 6; ++l) {
      const real ave =
          0.5 * (tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->bv(idx[0], idx[1], idx[2], l));
      zone->bv(idx[0], idx[1], idx[2], l) = ave;
      tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
    }
    for (int l = 0; l < zone->n_var; ++l) {
      const real ave =
          0.5 * (tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->cv(idx[0], idx[1], idx[2], l));
      zone->cv(idx[0], idx[1], idx[2], l) = ave;
      tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
    }
#if MULTISPECIES == 1
    for (int l = 0; l < zone->n_spec; ++l) {
      real ave = 0.5 * (tar_zone->yk(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->yk(idx[0], idx[1], idx[2], l));
      zone->yk(idx[0], idx[1], idx[2], l) = ave;
      tar_zone->yk(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
    }
#endif
  } else {
    // Else, get the inner value for this block's ghost grid
    for (int l = 0; l < 5; ++l) {
      zone->bv(idx[0], idx[1], idx[2], l) = tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l);
      zone->cv(idx[0], idx[1], idx[2], l) = tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l);
    }
    zone->bv(idx[0], idx[1], idx[2], 5) = tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], 5);
#if MULTISPECIES == 1
    for (int l = 0; l < zone->n_spec; ++l) {
      zone->yk(idx[0], idx[1], idx[2], l) = tar_zone->yk(idx_tar[0], idx_tar[1], idx_tar[2], l);
      zone->cv(idx[0], idx[1], idx[2], l + 5) = tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l + 5);
    }
#endif
  }
}

__global__ void cfd::eliminate_k_gradient(cfd::DZone *zone) {
  const integer ngg{zone->ngg}, mx{zone->mx}, my{zone->my};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  if (i >= mx + ngg || j >= my + ngg) return;

  auto &bv = zone->bv;
#if MULTISPECIES == 1
  auto &Y = zone->yk;
  const integer n_spec = zone->n_spec;
#endif

  for (integer k = 1; k <= ngg; ++k) {
    for (int l = 0; l < 6; ++l) {
      bv(i, j, k, l) = bv(i, j, 0, l);
      bv(i, j, -k, l) = bv(i, j, 0, l);
    }
#if MULTISPECIES == 1
    for (int l = 0; l < n_spec; ++l) {
      Y(i, j, k, l) = Y(i, j, 0, l);
      Y(i, j, -k, l) = Y(i, j, 0, l);
    }
#endif
  }
}
