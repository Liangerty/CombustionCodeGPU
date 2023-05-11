#include "Initialize.h"
#include "Define.h"
#include "Mesh.h"
#include <fstream>
#include "gxl_lib/MyString.h"
#include "BoundCond.h"
#include "fmt/format.h"

void cfd::initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, ChemData &chem_data) {
  const integer init_method = parameter.get_int("initial");
  switch (init_method) {
    case 0:
      initialize_from_start(parameter, mesh, field, chem_data);
      break;
    case 1:
      read_flowfield(parameter, mesh, field, chem_data);
      break;
    default:
      fmt::print("The initialization method is unknown, use freestream value to intialize by default.\n");
      initialize_from_start(parameter, mesh, field, chem_data);
  }
}

void cfd::initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, ChemData &chem_data) {
  // First read the initialization file to see if some patches are needed.
  std::ifstream init_file("input_files/setup/8_initialization.txt");
  std::string input{}, key{};
  int group{0}, tot_group{1};
  std::vector<Inflow> groups_inflow;
  std::vector<real> xs{0}, xe{0}, ys{0}, ye{0}, zs{0}, ze{0};
  std::istringstream line(input);
  while (gxl::getline_to_stream(init_file, input, line, gxl::Case::lower)) {
    line >> key;
    if (key == "//") continue;
    if (key == "int") {
      line >> key >> key >> tot_group;
      break;
    }
  }

  gxl::read_until(init_file, input, "label", gxl::Case::lower);
  while (group < tot_group) {
    if (input.rfind("end", 0) == 0) break; //input.starts_with("end")
    gxl::to_stringstream(input, line);
    int label{0};
    line >> key >> label;
    Inflow this_cond(label);

    bool found{false};
    std::string input2{};
    std::ifstream inflow_file("./input_files/setup/6_inflow.txt");
    while (!found) {
      gxl::read_until(inflow_file, input2, "label", gxl::Case::lower);
      if (input2.rfind("end", 0) == 0) break; //input2.starts_with("end")
      gxl::to_stringstream(input2, line);
      int label2{0};
      line >> key >> label2;
      if (label2 == label) {
        this_cond.register_boundary_condition(inflow_file, parameter, chem_data.spec);
        found = true;
      }
    }
    groups_inflow.push_back(this_cond);
    ++group;

    if (group > 1) {
      real xx{0}, yy{0};
      gxl::getline_to_stream(init_file, input, line);
      line >> key >> key >> key >> xx >> yy;
      xs.push_back(xx);
      xe.push_back(yy);
      gxl::getline_to_stream(init_file, input, line);
      line >> key >> key >> key >> xx >> yy;
      ys.push_back(xx);
      ye.push_back(yy);
      gxl::getline_to_stream(init_file, input, line);
      line >> key >> key >> key >> xx >> yy;
      zs.push_back(xx);
      ze.push_back(yy);
    }
    // Normally, there should be a set of reference values for computing the postprocess variables such as cf and qw, etc.
    // But I consider moving all postprocesses to a Postprocess executable, thus I would not set these up currently.
    //else {
    //  // This is the 1st group of inflow. Set as reference values by default.
    //  this_cond.set_as_reference(parameter);
    //}
    gxl::read_until(init_file, input, "label", gxl::Case::lower);
  }

  // Start to initialize
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    field[blk].initialize_basic_variables(parameter, groups_inflow, xs, xe, ys, ye, zs, ze);
  }


  if (parameter.get_int("myid") == 0) {
    fmt::print("Flowfield is initialized from given inflow conditions.\n");
    std::ofstream history("history.dat", std::ios::trunc);
    history << fmt::format("step\terror_max\n");
    history.close();
  }
}

void cfd::read_flowfield(cfd::Parameter &parameter, const cfd::Mesh &mesh, std::vector<Field> &field,
                         cfd::ChemData &chem_data) {

}
