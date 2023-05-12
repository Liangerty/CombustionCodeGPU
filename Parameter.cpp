#include "Parameter.h"
#include <fstream>
#include <sstream>
#include "Parallel.h"
#include <filesystem>

cfd::Parameter::Parameter(const MpiParallel &mpi_parallel) {
  read_param_from_file();
  int_parameters["myid"] = mpi_parallel.my_id;
  //int_parameters["n_proc"] = mpi_parallel.n_proc; // Currently commented, assuming the n_proc is not needed outside the class MpiParallel
  bool_parameters["parallel"] = cfd::MpiParallel::parallel;

  // Used for continue computing, record some info about the current simulation
//  const std::filesystem::path out_dir("output/message");
//  if (!exists(out_dir))
//    create_directories(out_dir);
//  std::ofstream ngg_out(out_dir.string()+"/ngg.txt");
//  ngg_out<<get_int("ngg");
//  ngg_out.close();
}

cfd::Parameter::Parameter(const std::string &filename) {
  std::ifstream file(filename);
  read_one_file(file);
  file.close();
}

//real cfd::Parameter::find_real(const std::string &name) const {
//  if (real_parameters.contains(name)) return real_parameters.at(name);
//  return 0;
//}

void cfd::Parameter::read_param_from_file() {
  for (auto &name: file_names) {
    std::ifstream file(name);
    read_one_file(file);
    file.close();
  }

  int_parameters.emplace("step",0);

  int_parameters.emplace("ngg", 2);
  integer inviscid_tag = get_int("inviscid_scheme");
  if (inviscid_tag / 10 == 5) {
    update_parameter("ngg", 3);
  }

  update_parameter("n_var", 5);
}

void cfd::Parameter::read_one_file(std::ifstream &file) {
  std::string input{}, type{}, key{}, temp{};
  std::istringstream line(input);
  while (std::getline(file, input)) {
    if (input.starts_with("//") || input.starts_with("!") || input.empty()) {
      continue;
    }
    line.clear();
    line.str(input);
    line >> type;
    line >> key >> temp;
    if (type == "int") {
      int val{};
      line >> val;
      int_parameters.emplace(std::make_pair(key, val));
    } else if (type == "double") {
      real val{};
      line >> val;
      real_parameters.emplace(std::make_pair(key, val));
    } else if (type == "bool") {
      bool val{};
      line >> val;
      bool_parameters.emplace(std::make_pair(key, val));
    } else if (type == "string") {
      std::string val{};
      line >> val;
      string_parameters.emplace(std::make_pair(key, val));
    }
  }
  file.close();
}
