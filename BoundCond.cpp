#include "BoundCond.h"
#include "ChemData.h"
#include "gxl_lib/MyString.h"
#include "Transport.cuh"
#include <cmath>

cfd::Inflow::Inflow(integer type_label) : label(type_label) {}

std::tuple<real, real, real, real, real, real> cfd::Inflow::var_info() const {
  return std::make_tuple(density, u, v, w, pressure, temperature);
}

cfd::Inflow::Inflow(const std::map<std::string, std::variant<std::string, integer, real>> &info,
                    Species &spec) :
    label{std::get<integer>(info.at("label"))} {

  // In default, the mach number, pressure and temperature should be given.
  // If other combinations are given, then implement it later.
  // Currently, 2 combinations are achieved. One is to give (mach, pressure,
  // temperature) The other is to give (density, velocity, pressure)
  if (info.contains("mach")) mach = std::get<real>(info.at("mach"));
  if (info.contains("pressure")) pressure = std::get<real>(info.at("pressure"));
  if (info.contains("temperature")) temperature = std::get<real>(info.at("temperature"));
  if (info.contains("velocity")) velocity = std::get<real>(info.at("velocity"));
  if (info.contains("density")) density = std::get<real>(info.at("density"));
  if (info.contains("u")) u = std::get<real>(info.at("u"));
  if (info.contains("v")) v = std::get<real>(info.at("v"));
  if (info.contains("w")) w = std::get<real>(info.at("w"));
#if MULTISPECIES == 1
  const integer n_spec{spec.n_spec};
  yk = new real[n_spec];
  for (int qq = 0; qq < n_spec; ++qq) {
    yk[qq] = 0;
  }

  // Assign the species mass fraction to the corresponding position.
  // Should be done after knowing the order of species.
  for (auto [name, idx]: spec.spec_list) {
    if (info.contains(name)) {
      yk[idx] = std::get<real>(info.at(name));
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

#if MULTISPECIES == 1
  std::vector<real> cpi(n_spec, 0);
  spec.compute_cp(temperature, cpi.data());
  real cp{0}, cv{0};
  for (size_t i = 0; i < n_spec; ++i) {
    cp += yk[i] * cpi[i];
    cv += yk[i] * (cpi[i] - R_u / spec.mw[i]);
  }
  real gamma = cp / cv;  // specific heat ratio
#else
  real gamma{gamma_air};  // specific heat ratio
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

cfd::Wall::Wall(const std::map<std::string, std::variant<std::string, integer, real>> &info)
    : label(std::get<integer>(info.at("label"))) {
  if (info.contains("thermal_type")) {
    thermal_type = std::get<std::string>(info.at("thermal_type")) == "isothermal" ? ThermalType::isothermal : ThermalType::adiabatic;
  }
  if (thermal_type == ThermalType::isothermal) {
    if (info.contains("temperature")) {
      temperature = std::get<real>(info.at("temperature"));
    } else {
      printf("Isothermal wall does not specify wall temperature, is set as 300K in default.\n");
    }
  }
}

cfd::Outflow::Outflow(integer type_label) : label(type_label) {}
