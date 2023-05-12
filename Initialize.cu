#include "Initialize.h"
#include "Define.h"
#include "Mesh.h"
#include <fstream>
#include "gxl_lib/MyString.h"
#include "BoundCond.h"
#include "fmt/format.h"
#include <filesystem>

void cfd::initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field,
                                     ChemData &chem_data) {
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

void
cfd::initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, ChemData &chem_data) {
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
  const std::filesystem::path out_dir("output/field");
  if (!exists(out_dir)) {
    fmt::print("The directory to flowfield files does not exist!\n");
  }
  FILE *fp = fopen((out_dir.string() + fmt::format("/flowfield{:>4}.plt", parameter.get_int("myid"))).c_str(), "rb");

  std::string magic_number;
  fread(magic_number.data(), 8, 1, fp);
  int32_t byte_order{1};
  fread(&byte_order, 4, 1, fp);
  int32_t file_type{0};
  fread(&file_type, 4, 1, fp);
  std::string solution_file = gxl::read_str(fp);
  integer n_var_old{5};
  fread(&n_var_old, 4, 1, fp);
  std::vector<std::string> var_name;
  var_name.resize(n_var_old);
  for (size_t i = 0; i < n_var_old; ++i) {
    var_name[i] = gxl::read_str(fp);
  }
  auto index_order = cfd::identify_variable_labels(var_name, chem_data);
  const integer n_spec{chem_data.spec.n_spec};
#if MULTISPECIES==1
  constexpr integer index_spec_start{6};
  const integer index_spec_end{6+n_spec};
  bool has_spec_info{false};
  for (auto ii:index_order){
    if (ii>=index_spec_start&&ii<index_spec_end)
      has_spec_info=true;
  }
#endif

  float marker{0.0f};
  constexpr float eohmarker{357.0f};
  fread(&marker, 4, 1, fp);
  std::vector<std::string> zone_name;
  std::vector<double> solution_time;
  integer zone_number{0};
  while (fabs(marker - eohmarker) > 1e-25f) {
    zone_name.emplace_back(gxl::read_str(fp));
    int32_t parent_zone{-1};
    fread(&parent_zone, 4, 1, fp);
    int32_t strand_id{-2};
    fread(&strand_id, 4, 1, fp);
    real sol_time{0};
    fread(&sol_time, 8, 1, fp);
    solution_time.emplace_back(sol_time);
    int32_t zone_color{-1};
    fread(&zone_color, 4, 1, fp);
    int32_t zone_type{0};
    fread(&zone_type, 4, 1, fp);
    int32_t var_location{0};
    fread(&var_location, 4, 1, fp);
    int32_t raw_face_neighbor{0};
    fread(&raw_face_neighbor, 4, 1, fp);
    int32_t miscellaneous_face{0};
    fread(&miscellaneous_face, 4, 1, fp);
    integer mx{0}, my{0}, mz{0};
    fread(&mx, 4, 1, fp);
    fread(&my, 4, 1, fp);
    fread(&mz, 4, 1, fp);
    int32_t auxi_data{1};
    fread(&auxi_data, 4, 1, fp);
    while (auxi_data != 0) {
      auto auxi_name{gxl::read_str(fp)};
      int32_t auxi_format{0};
      fread(&auxi_format, 4, 1, fp);
      auto auxi_val{gxl::read_str(fp)};
      if (auxi_name == "step") {
        parameter.update_parameter("step", std::stoi(auxi_val));
      }
      fread(&auxi_data, 4, 1, fp);
    }
    ++zone_number;
    fread(&marker, 4, 1, fp);
  }

  // Next, data section
  for (size_t b = 0; b < mesh.n_block; ++b) {
    fread(&marker, 4, 1, fp);
    int32_t data_format{1};
    for (int l = 0; l < n_var_old; ++l) {
      fread(&data_format, 4, 1, fp);
    }
    size_t data_size{4};
    if (data_format == 2) {
      data_size = 8;
    }
    int32_t passive_var{0};
    fread(&passive_var, 4, 1, fp);
    int32_t shared_var{0};
    fread(&shared_var, 4, 1, fp);
    int32_t shared_connect{-1};
    fread(&shared_connect, 4, 1, fp);
    double max{0}, min{0};
    for (int l = 0; l < n_var_old; ++l) {
      fread(&min, 8, 1, fp);
      fread(&max, 8, 1, fp);
    }
    // zone data
    // First, the coordinates x, y and z.
    const integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    for (size_t l = 0; l < 3; ++l) {
      read_one_useless_variable(fp, mx, my, mz, data_format);
    }

    // Other variables
    for (size_t l = 3; l < n_var_old; ++l) {
      auto index = index_order[l];
      if (index < 6) {
        // basic variables
        auto &bv = field[b].h_zone.bv;
        if (data_format == 1) {
          // float storage
          float v{0.0f};
          for (int k = 0; k < mz; ++k) {
            for (int j = 0; j < my; ++j) {
              for (int i = 0; i < mx; ++i) {
                fread(&v, data_size, 1, fp);
                bv(i, j, k, index) = v;
              }
            }
          }
        } else {
          // double storage
          for (int k = 0; k < mz; ++k) {
            for (int j = 0; j < my; ++j) {
              for (int i = 0; i < mx; ++i) {
                fread(&bv(i, j, k, index), data_size, 1, fp);
              }
            }
          }

        }
      }
#if MULTISPECIES == 1
      else if (index < 6 + n_spec) {
        // species variables
        auto &yk = field[b].h_zone.yk;
        index -= 6;
        if (data_format == 1) {
          // float storage
          float v{0.0f};
          for (int k = 0; k < mz; ++k) {
            for (int j = 0; j < my; ++j) {
              for (int i = 0; i < mx; ++i) {
                fread(&v, data_size, 1, fp);
                yk(i, j, k, index) = v;
              }
            }
          }
        } else {
          // double storage
          for (int k = 0; k < mz; ++k) {
            for (int j = 0; j < my; ++j) {
              for (int i = 0; i < mx; ++i) {
                fread(&yk(i, j, k, index), data_size, 1, fp);
              }
            }
          }

        }
      }
#endif
      else if (index == 6 + n_spec) {
        // Other variables, such as, turbulent variables et, al.
        // Each one has a independent label...
      } else {
        // No matched label, just ignore
        read_one_useless_variable(fp, mx, my, mz, data_format);
      }
    }
  }

  // Next, if the previous simulation does not contain some of the variables used in the current simulation,
  // then we intialize them here
#if MULTISPECIES==1
  if (!has_spec_info){
    initialize_spec_from_inflow(parameter, mesh, field, chem_data);
  }
#endif

  if (parameter.get_int("myid") == 0) {
    fmt::print("Flowfield is initialized from previous simulation results.\n");
  }
}

std::vector<integer> cfd::identify_variable_labels(std::vector<std::string> &var_name, ChemData &chem_data) {
  std::vector<integer> labels;
  const auto &spec_name = chem_data.spec.spec_list;
  const integer n_spec = chem_data.spec.n_spec;
  for (auto &name: var_name) {
    integer l = 999;
    // The first three names are x, y and z, they are assigned value 0 and no match would be found.
    auto n = gxl::to_upper(name);
    if (n == "DENSITY" || n == "ROE" || n == "RHO") {
      l = 0;
    } else if (n == "U") {
      l = 1;
    } else if (n == "V") {
      l = 2;
    } else if (n == "W") {
      l = 3;
    } else if (n == "P"||n=="PRESSURE") {
      l = 4;
    } else if (n == "T"||n=="TEMPERATURE") {
      l = 5;
    } else if (n == "MUT") { // To be determined
      l = 6 + n_spec;
    } else {
      for (auto [spec, sp_label]: spec_name) {
        if (n == gxl::to_upper(spec)) {
          l = 6 + sp_label;
          break;
        }
      }
    }
    labels.emplace_back(l);
  }
  return labels;
}

void cfd::read_one_useless_variable(FILE *fp, integer mx, integer my, integer mz, integer data_format) {
  if (data_format == 1) {
    // float
    float v{0.0f};
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          fread(&v, 4, 1, fp);
        }
      }
    }
  } else {
    // double
    double v{0.0};
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          fread(&v, 8, 1, fp);
        }
      }
    }
  }
}

#if MULTISPECIES==1
void cfd::initialize_spec_from_inflow(cfd::Parameter &parameter, const cfd::Mesh &mesh, std::vector<Field> &field,
                                      cfd::ChemData &chem_data) {
  // This can also be implemented like the from_start one, which can have patch.
  // But currently, for easy to implement, just intialize the whole flowfield to the inflow composition,
  // which means that other species would have to be computed from boundary conditions.
  // If the need for initialize species in groups is strong,
  // then we implement it just by copying the previous function "initialize_from_start",
  // which should be easy.
  bool found{false};
  std::string input2{}, key{};
  std::istringstream line(input2);
  std::ifstream inflow_file("./input_files/setup/6_inflow.txt");
  constexpr integer label{5};
  Inflow inflow(label);
  while (!found) {
    gxl::read_until(inflow_file, input2, "label", gxl::Case::lower);
    if (input2.rfind("end", 0) == 0) break; //input2.starts_with("end")
    gxl::to_stringstream(input2, line);
    int label2{0};
    line >> key >> label2;
    if (label2 == label) {
      inflow.register_boundary_condition(inflow_file, parameter, chem_data.spec);
      found = true;
    }
  }
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const integer mx{mesh[blk].mx},my{mesh[blk].my},mz{mesh[blk].mz};
    const auto n_spec = parameter.get_int("n_spec");
    auto mass_frac= inflow.yk;
    auto& yk=field[blk].h_zone.yk;
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (int l = 0; l < n_spec; ++l) {
            yk(i, j, k, l) = mass_frac[l];
          }
        }
      }
    }
  }
  if (parameter.get_int("myid")==0)
    fmt::print("Compute from single species result. The species field is initialized with freestream.\n");
}
#endif