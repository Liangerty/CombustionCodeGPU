#pragma once

/**
 * \brief The physical constants used in our simulation
 *
 * \ref [NIST Reference on Constants, Units and Uncertanity](https://physics.nist.gov/cuu/Reference/contents.html)
 *
 **/
namespace cfd {
// Avogadro's Number [kmole^{-1}]
constexpr double avogadro = 6.02214076e26;
// Elementary charge  [C]
constexpr double electron_charge = 1.602176634e-19;
// Electron Mass  [kg]
constexpr double electron_mass = 9.1093837015e-31;
// Universal gas constant [J/(kmole*K)]
constexpr double R_u = 8314.462618;
// Universal gas constant [cal/(mole*K)], fetched from [Gas constant](https://en.wikipedia.org/wiki/Gas_constant)
constexpr double R_c = 1.98720425864083;
// Specific heat ratio for air in perfect gas
constexpr double gamma_air = 1.4;
// Atmospheric pressure
constexpr double p_atm = 101325;
// Air molecular weight.
// Ref: Engineering ToolBox, (2003). Air - Thermophysical Properties. [online] Available at: https://www.engineeringtoolbox.com/air-properties-d_156.html [Accessed Day Mo. Year].
constexpr double mw_air = 28.9647;

// Some model constants
struct SST{
  static constexpr double beta_star=0.09;
  static constexpr double sqrt_beta_star=0.3;
  static constexpr double kappa=0.41;
  // SST inner parameters, the first group:
  static constexpr double sigma_k1=0.85;
  static constexpr double sigma_omega1=0.5;
  static constexpr double beta_1=0.0750;
  static constexpr double a_1=0.31;
  static constexpr double gamma1=beta_1/beta_star-sigma_omega1*kappa*kappa/sqrt_beta_star;

  // k-epsilon parameters, the second group:
  static constexpr double sigma_k2=1;
  static constexpr double sigma_omega2=0.856;
  static constexpr double beta_2=0.0828;
  static constexpr double gamma2=beta_2/beta_star-sigma_omega2*kappa*kappa/sqrt_beta_star;

  // Mixed parameters, their difference, used in computations
  static constexpr double delta_sigma_k=sigma_k1-sigma_k2;
  static constexpr double delta_sigma_omega=sigma_omega1-sigma_omega2;
  static constexpr double delta_beta=beta_1-beta_2;
  static constexpr double delta_gamma=gamma1-gamma2;
};
}
