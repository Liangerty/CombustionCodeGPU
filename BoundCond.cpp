#include "BoundCond.h"
#include "ChemData.h"
#include "Define.h"
#include "gxl_lib/MyString.h"
#include "Transport.cuh"
#include <cmath>

cfd::Inflow::Inflow(integer type_label) : label(type_label) {}

void cfd::Inflow::register_boundary_condition(std::ifstream &file,
                                              Parameter &parameter
#if MULTISPECIES == 1
    , Species &spec
#endif
) {
  std::map<std::string, real> par;
  std::map<std::string, real> spec_inflow;
  std::string input{}, key{}, name{};
  real val{};
  std::istringstream line(input);
  while (gxl::getline_to_stream(file, input, line, gxl::Case::lower)) {
    line >> key;
    if (key == "double") {
      line >> name >> key >> val;
      par.emplace(std::make_pair(name, val));
      continue;
    }
    if (key == "species") {
      line >> name >> key >> val;
      spec_inflow.emplace(std::make_pair(gxl::to_upper(name), val));
      continue;
    }
    if (key == "label" || key == "end") break;
  }
  // In default, the mach number, pressure and temperature should be given.
  // If other combinations are given, then implement it later.
  // Currently, 2 combinations are achieved. One is to give (mach, pressure,
  // temperature) The other is to give (density, velocity, pressure)
  if (par.contains("mach")) mach = par["mach"];
  if (par.contains("pressure")) pressure = par["pressure"];
  if (par.contains("temperature")) temperature = par["temperature"];
  if (par.contains("velocity")) velocity = par["velocity"];
  if (par.contains("density")) density = par["density"];
  if (par.contains("u")) u = par["u"];
  if (par.contains("v")) v = par["v"];
  if (par.contains("w")) w = par["w"];
#if MULTISPECIES == 1
  const integer n_spec{parameter.get_int("n_spec")};
  yk = new real[n_spec];
  for (int qq = 0; qq < n_spec; ++qq) {
    yk[qq] = 0;
  }

  // Assign the species mass fraction to the corresponding position.
  // Should be done after knowing the order of species.
  for (auto &[sp, y]: spec_inflow) {
    if (spec.spec_list.contains(sp)) {
      const integer idx = spec.spec_list.at(sp);
      yk[idx] = y;
    }
  }
  mw = 0;
  for (int i = 0; i < n_spec; ++i) mw += yk[i] / spec.mw[i];
  mw = 1 / mw;
#endif  // MULTISPECIES==1

  if (temperature < 0) {
    // The temperature is not given, thus the density and pressure are given
    temperature = pressure * mw / (density * R_u);
  }
  viscosity = Sutherland(temperature);

  real gamma{gamma_air};  // specific heat ratio
  real cp{0}, cv{0};
#if MULTISPECIES == 1
  std::vector<real> cpi(n_spec, 0);
  spec.compute_cp(temperature, cpi.data());
  for (size_t i = 0; i < n_spec; ++i) {
    cp += yk[i] * cpi[i];
    cv += yk[i] * (cpi[i] - R_u / spec.mw[i]);
  }
  gamma = cp / cv;
#endif  // MULTISPECIES==1

  const real c{std::sqrt(gamma * R_u / mw * temperature)};  // speed of sound

  if (mach < 0) {
    // The mach number is not given. The velocity magnitude should be given
    mach = velocity / c;
  } else {
    // The velocity magnitude is not given. The mach number should be given
    velocity = mach * c;
  }
  u *= velocity;
  v *= velocity;
  w *= velocity;
  if (density < 0) {
    // The density is not given, compute it from equation of state
    density = pressure * mw / (R_u * temperature);
  }
#if MULTISPECIES == 1
  viscosity = compute_viscosity(temperature, mw, yk, spec);
#endif  // MULTISPECIES==1
//  reynolds =
//      density * velocity * parameter.get_real("referenceLength") / viscosity;
}

std::tuple<real, real, real, real, real, real> cfd::Inflow::var_info() const {
  return std::make_tuple(density, u, v, w, pressure, temperature);
}

cfd::Wall::Wall(integer type_label, std::ifstream &bc_file) : label(type_label) {
  std::map<std::string, std::string> opt;
  std::map<std::string, double> par;
  std::string input{}, key{}, name{};
  double val{};
  std::istringstream line(input);
  while (gxl::getline_to_stream(bc_file, input, line, gxl::Case::lower)) {
    line >> key;
    if (key == "double") {
      line >> name >> key >> val;
      par.emplace(std::make_pair(name, val));
    } else if (key == "option") {
      line >> name >> key >> key;
      opt.emplace(std::make_pair(name, key));
    }
    if (key == "label" || key == "end") {
      break;
    }
  }
  if (opt.contains("thermal_type")) {
    thermal_type = opt["thermal_type"] == "isothermal" ? ThermalType::isothermal : ThermalType::adiabatic;
  }
  if (thermal_type == ThermalType::isothermal) {
    if (par.contains("temperature")) {
      temperature = par["temperature"];
    } else {
      printf("Isothermal wall does not specify wall temperature, is set as 300K in default.\n");
    }
  }
}

cfd::Outflow::Outflow(integer type_label) : label(type_label) {}
