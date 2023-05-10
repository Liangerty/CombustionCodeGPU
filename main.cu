#include <cstdio>
#include "Parallel.h"
#include "Parameter.h"
#include "Mesh.h"
#include "ChemData.h"
#include "Driver.h"
// #include "Model.h"
// #include "Field.h"
// #include <cuda_runtime.h>
// #include "Driver.h"
// #include "Zone.h"
// #include "ViscousScheme.hpp"

// __global__ void print_message() {
//   printf("OK\n");
// }

int main(int argc, char *argv[]) {
  cfd::MpiParallel mpi_parallel(&argc, &argv);

  // Set the heap maximum value for kernel functions, must be set before any cuda functions called.
  // In the multispecies simulation, currently the requested memory is limited by the number of species.
  // Now set 1GB for this value. When the grid number or species number increases, this value may need to be increased.
    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 100*1024*1024);

  cfd::Parameter parameter(mpi_parallel);

  cfd::Mesh mesh(parameter);

#if MULTISPECIES == 1
  cfd::ChemData chem_data(parameter);
#else
  printf("Air computation.\n");
#endif

  cfd::Driver driver(parameter, mesh
#if MULTISPECIES == 1
      , chem_data
#endif
  );

  driver.initialize_computation();

  driver.simulate();

  printf("Fuck off\n");
  return 0;

  //

  //
  // //printf("x(5,3,0) = %e\n", mesh[0].x(5, 3, 0));

  // cfd::Model model(parameter);

  // cfd::Field field(parameter, mesh, model);

  // gcfd::Driver driver(parameter,mesh,model,field);
  // gcfd::call<<<1,10>>>(driver.viscous_scheme);

  //int i{0};
  //scanf("%d", &i);
  //gcfd::compute_viscous_flux<0, 2,4,6,8,10>(i);

  //gcfd::print_info<<<1,1>>>(&driver.zone[0].d_ptr[0]);

  // h_data与d_data都是GPU上的数据，但是h_data位于CPU端，包含的指针指向GPU端内存；d_data则完全位于GPU端，可在核函数中使用
  // gcfd::DevData* h_data = new gcfd::DevData(parameter, mesh, model);
  // gcfd::DevData* d_data = nullptr;
  // cudaMalloc(&d_data, sizeof(gcfd::DevData));
  // gcfd::copy_to_gpu(d_data,h_data);
  // cudaMemcpy(d_data, h_data, sizeof(gcfd::DevData), cudaMemcpyHostToDevice);




  //auto err = cudaMemcpy(driver.zone[0].h_ptr->bv.data(), field.zones[0].bv.data(), sizeof(real), cudaMemcpyHostToDevice);
  //if (err != cudaSuccess) {
  //  printf("Error %d: \"%s\" in %s at line %d\n", int(err), cudaGetErrorString(err), __FILE__, __LINE__);
  //}
  //err=cudaMemcpy(field.zones[0].bv.data(), driver.zone[0].h_ptr->bv.data(), sizeof(real), cudaMemcpyDeviceToHost);
  //if (err != cudaSuccess) {
  //  printf( "Error %d: \"%s\" in %s at line %d\n", int(err), cudaGetErrorString(err), __FILE__, __LINE__);
  //}
  // print_message << <10, 1 >> > ();
  //if (err != cudaSuccess) {
  //  fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err), cudaGetErrorString(err), __FILE__, __LINE__);
  //}
}
