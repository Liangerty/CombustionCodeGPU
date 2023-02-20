#pragma once

#include "Define.h"
#include "gxl_lib/Array.hpp"
#include "Parameter.h"
#include "Mesh.h"

namespace cfd {
//class Block;

struct Inflow;

struct HZone {
  HZone(Parameter &parameter, const Block &block);

  void initialize_basic_variables(const Parameter &parameter, const Block &block, const std::vector<Inflow> &inflows,
                                  const std::vector<real> &xs, const std::vector<real> &xe, const std::vector<real> &ys,
                                  const std::vector<real> &ye, const std::vector<real> &zs,
                                  const std::vector<real> &ze);

  gxl::VectorField3D<real> cv;
  gxl::VectorField3D<real> bv;
  gxl::Array3D<real> mach;
#if MULTISPECIES == 1
  gxl::VectorField3D<real> yk;
#endif
#ifdef DEBUG
  gxl::VectorField3D<real> dbv_squared;
  gxl::Array3D<real> tempo_var;
  gxl::VectorField3D<real> dq;
#endif
};

#ifdef GPU

struct DZone {
  DZone() = default;

  integer mx = 0, my = 0, mz = 0, ngg = 0, n_spec = 0, n_scal = 0, n_var = 5;
  ggxl::Array3D<real> x, y, z;
  Boundary *boundary= nullptr;
  InnerFace *innerface= nullptr;
  ParallelFace *parface= nullptr;
  ggxl::Array3D<real> jac;
  ggxl::Array3D<gxl::Matrix<real, 3, 3, 1>> metric;
  ggxl::VectorField3D<real> cv; // Conservative variable: 0-:rho, 1-:rho*u, 2-:rho*v, 3-:rho*w, 4-:rho*(E+V*V/2), 5->(4+Ns)-:rho*Y
  ggxl::VectorField3D<real> bv; // Basic variable: 0-density, 1-u, 2-v, 3-w, 4-pressure, 5-temperature
  ggxl::VectorField3D<real> bv_last; // Basic variable of last step: 0-density, 1-velocity magnitude, 2-pressure, 3-temperature
  ggxl::Array3D<real> vel;      // Velocity magnitude
  ggxl::Array3D<real> acoustic_speed;
  ggxl::Array3D<real> mach;     // Mach number
  ggxl::Array3D<real> mul;      // Dynamic viscosity
  ggxl::Array3D<real> conductivity;      // Dynamic viscosity
#if MULTISPECIES == 1
  ggxl::VectorField3D<real> yk; // Mass fraction of various species
  ggxl::VectorField3D<real> rho_D; // the mass diffusivity of species
  ggxl::Array3D<real> gamma;  // specific heat ratio
#endif // MULTISPECIES==1
  ggxl::VectorField3D<real> dq; // The residual for flux computing
  ggxl::Array3D<real[3]> inv_spectr_rad;  // inviscid spectral radius. Used when LUSGS type temporal scheme is used.
  ggxl::Array3D<real> visc_spectr_rad;  // viscous spectral radius.
  ggxl::Array3D<real> dt_local; //local time step. Used for steady flow simulation
};

#endif

struct Field {
  Field(Parameter &parameter, const Block &block);

  void initialize_basic_variables(const Parameter &parameter, const Block &block, const std::vector<Inflow> &inflows,
                                  const std::vector<real> &xs, const std::vector<real> &xe, const std::vector<real> &ys,
                                  const std::vector<real> &ye, const std::vector<real> &zs,
                                  const std::vector<real> &ze);

  void setup_device_memory(const Parameter &parameter, const Block &block);

  HZone h_zone;
#ifdef GPU
  DZone *d_ptr = nullptr;
  DZone *h_ptr = nullptr;
#endif
};

struct DParameter;

__global__ void compute_cv_from_bv(DZone *zone, DParameter* param);

__global__ void update_physical_properties(DZone *zone, DParameter* param);

__global__ void inner_communication(DZone *zone, DZone *tar_zone, const uint n_point[3], integer i_face);

__global__ void eliminate_k_gradient(DZone *zone);
}