#pragma once

#include "Parameter.h"
#include "Define.h"
#include "gxl_lib/Matrix.hpp"

namespace cfd {
struct Species;
struct Reaction;
struct DParameter {
  DParameter()=default;
  explicit DParameter(Parameter &parameter,Species& species, Reaction& reaction);

//  integer myid = 0;   // The process id of this process
//  integer dim = 3;  // The dimension of the simulation problem
//  integer n_block=1;  // number of blocks in this process
  integer inviscid_scheme=0;  // The tag for inviscid scheme. 3 - AUSM+
  integer reconstruction=2; // The reconstruction method for inviscid flux computation
  integer limiter=0;  // The tag for limiter method
  integer viscous_scheme=0; // The tag for viscous scheme. 0 - Inviscid, 2 - 2nd order central discretization
  integer rans_model=0;  // The tag for RANS model. 0 - Laminar, 1 - SA, 2 - SST
  integer turb_implicit = 1;    // If we implicitly treat the turbulent source term. By default, implicitly treat(1), else, 0(explicit)
  integer implicit_method = 0;  // The tag for implicit treatment method. 0 - explicit, 1 - DPLUR
  integer DPLUR_inner_step = 2; // If we use DPLUR, then we need a specified number of inner iterations.
  integer chemSrcMethod = 0;  // For finite rate chemistry, we need to know how to implicitly treat the chemical source
//  integer output_screen=10; // determine the interval between screen outputs
  real Pr=0.72;
  real cfl=1;
  integer n_spec=0;
  real* mw = nullptr;
  ggxl::MatrixDyn<real> high_temp_coeff, low_temp_coeff;
  real* t_low = nullptr, * t_mid = nullptr, * t_high = nullptr;
  real* LJ_potent_inv = nullptr;
  real* vis_coeff = nullptr;
  ggxl::MatrixDyn<real> WjDivWi_to_One4th;
  ggxl::MatrixDyn<real> sqrt_WiDivWjPl1Mul8;
  real Sc=0.9;
  real Prt=0.9;
  real Sct=0.9;
};
}
