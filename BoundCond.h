#pragma once

#include "Constants.h"
#include "Define.h"
#include "Parameter.h"

#if MULTISPECIES == 1

#include "ChemData.h"

#endif

namespace cfd {
struct Inflow {
  explicit Inflow(integer type_label);

#ifdef GPU

  Inflow(integer type_label, std::ifstream &file
#if MULTISPECIES == 1
      , Species &spec
#endif
  );

#endif

  void register_boundary_condition(std::ifstream &file, Parameter &parameter
#if MULTISPECIES == 1
      , Species &spec
#endif
  );

  [[nodiscard]] std::tuple<real, real, real, real, real, real> var_info() const;

  integer label = 5;
  real mach = -1;
  real pressure = 101325;
  real temperature = -1;
  real velocity = 0;
  real density = -1;
  real u = 1, v = 0, w = 0;
#if MULTISPECIES == 1
  real *yk = nullptr;
#endif
  real mw = mw_air;
  real viscosity = 0;
//  real reynolds=0;
};

struct Wall {
  explicit Wall(integer type_label, std::ifstream &bc_file);

  enum class ThermalType { isothermal, adiabatic };

  integer label = 2;
  ThermalType thermal_type = ThermalType::isothermal;
  real temperature{300};
};

struct Outflow {
  explicit Outflow(integer type_label);

  integer label = 6;
};
}
