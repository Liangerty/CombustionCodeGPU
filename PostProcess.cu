#include "PostProcess.h"

//#include <filesystem>
//#include <fmt/format.h>
//#include "Mesh.h"
//#include "Solver.h"

//cfd::PostProcess::PostProcess(const Mesh& _mesh, const Solver& solver, const Parameter& _parameter):
//  parameter{_parameter}, mesh{_mesh}, field{std::span{solver.field}} {
//  // functions.emplace_back(wall_friction_heatflux_2d);
//}

//void cfd::PostProcess::operator()() const {
//  for (auto& fun : functions)
//    fun(mesh, field, parameter);
//}

//void cfd::wall_friction_heatflux_2d(const Mesh& mesh, const std::span<const Variable> field,
//                                      const Parameter& parameter) {
//  const std::filesystem::path out_dir("output/wall");
//  if (!exists(out_dir))
//    create_directories(out_dir);
//  const auto path_name = out_dir.string();
//
//  int size{mesh[0].mx};
//  for (int blk = 1; blk < mesh.n_block; ++blk) {
//    if (mesh[blk].mx > size)
//      size = mesh[blk].mx;
//  }
//  std::vector<double> friction(size, 0);
//  std::vector<double> heat_flux(size, 0);
//
//  const double rho_inf = parameter.get_real("rho_inf");
//  const double v_inf   = parameter.get_real("v_inf");
//  // const double t_inf = parameter.get_double("t_inf");
//  const double dyn_pressure = 0.5 * rho_inf * v_inf * v_inf;
//  for (int blk = 0; blk < mesh.n_block; ++blk) {
//    auto& block = mesh[blk];
//    const int mx{block.mx};
//    for (int i = 0; i < mx; ++i) {
//      std::span pv{field[blk].basic_var(i, 1, 0), 7};
//      auto& metric       = block.metric(i, 0, 0);
//      const double xi_x  = metric(1, 1), xi_y  = metric(1, 2);
//      const double eta_x = metric(2, 1), eta_y = metric(2, 2);
//
//      const double viscosity       = field[blk].viscosity(i, 0, 0);
//      const double u_parallel_wall = (xi_x * pv[1] + xi_y * pv[2]) / sqrt(xi_x * xi_x + xi_y * xi_y);
//      const double grad_eta        = sqrt(eta_x * eta_x + eta_y * eta_y);
//      const double du_normal_wall  = u_parallel_wall * grad_eta;
//      // dimensionless fiction coefficient, cf
//      friction[i] = viscosity * du_normal_wall / dyn_pressure;
//
//      const double conductivity = field[blk].conductivity(i, 0, 0);
//      // dimensional heat flux
//      heat_flux[i] = conductivity * (pv[5] - field[blk].basic_var(i, 0, 0, 5)) * grad_eta;
//    }
//
//    std::ofstream f(path_name + fmt::format("/friction_heatflux {} {}.dat", parameter.get_int("myid"), blk));
//    f << fmt::format("variables = \"x\", \"cf\", \"qw\"\n");
//    for (int i = 0; i < mx; ++i)
//      f << fmt::format("{}\t{:e}\t{:e}\n", block.x(i, 0, 0), friction[i], heat_flux[i]);
//    f.close();
//  }
//}


__global__ void cfd::wall_friction_heatFlux_2d(cfd::DZone *zone, real *friction, real *heat_flux, real dyn_pressure) {
  const integer i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=zone->mx) return;

  auto &pv = zone->bv;

  auto &metric = zone->metric(i, 0, 0);
  const real xi_x = metric(1, 1), xi_y = metric(1, 2);
  const real eta_x = metric(2, 1), eta_y = metric(2, 2);

  const real viscosity = zone->mul(i, 0, 0);
  const double u_parallel_wall = (xi_x * pv(i, 1, 0, 1) + xi_y * pv(i, 1, 0, 2)) / sqrt(xi_x * xi_x + xi_y * xi_y);
  const double grad_eta = sqrt(eta_x * eta_x + eta_y * eta_y);
  const double du_normal_wall = u_parallel_wall * grad_eta;
  // dimensionless fiction coefficient, cf
  friction[i] = viscosity * du_normal_wall / dyn_pressure;

  const double conductivity = zone->thermal_conductivity(i, 0, 0);
  // dimensional heat flux
  heat_flux[i] = conductivity * (pv(i, 1, 0, 5) - pv(i, 0, 0, 5)) * grad_eta;

}
