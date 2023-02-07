#pragma once
#ifdef __CUDACC__

#include <cuda_runtime.h>

#endif

#include <vector>

#ifdef __CUDACC__
namespace ggxl {
template<typename T>
class Array3D {
private:
  int ng = 0;
  int n1 = 0, n2 = 0, n3 = 0;
  int disp1 = 0, disp2 = 0, dispt = 0;
  int sz = 0;
  T *val = nullptr;

public:
  Array3D() {}

  // Shallow copy constructor
  Array3D(const Array3D& b):ng(b.ng), n1(b.n1), n2(b.n2),n3(b.n3),disp1(b.disp1),disp2(b.disp2),dispt(b.dispt),sz(b.sz),val(b.val){}

//  Array3D(int dim1, int dim2, int dim3, int n_ghost = 0)
//      : ng(n_ghost),
//        n1(dim1),
//        n2(dim2),
//        n3(dim3),
//        disp2{n3 + 2 * ng},
//        disp1{(n2 + 2 * ng) * disp2},
//        dispt{(disp1 + disp2 + 1) * ng},
//        sz{(n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng)} {
//    cudaMalloc(&val, sz * sizeof(T));
//  }

//   void resize(integer dim1, integer dim2, integer dim3, integer n_ghost=0);

  cudaError_t allocate_memory(integer dim1, integer dim2, integer dim3, integer n_ghost = 0);

   __device__ T& operator()(const int i, const int j, const int k) {
     return val[i * disp1 + j * disp2 + k + dispt];
   }
   __device__ const T& operator()(const int i, const int j, const int k) const {
     return val[i * disp1 + j * disp2 + k + dispt];
   }

  T *data() { return val; }

  const T *data() const { return val; }

  auto size(){return sz;}

//   __device__ void print_info();

//   CUDA_CALLABLE_MEMBER ~Array3D() { /*cudaFree(val);*/ }
};

// template <typename T>
// inline void Array3D<T>::resize(integer dim1, integer dim2, integer dim3, integer n_ghost){
//   ng=n_ghost;
//   n1=dim1;n2=dim2;n3=dim3;
//   disp2=n3+2*ng;
//   disp1=(n2+2*ng)*disp2;
//   dispt=(disp1+disp2+1)*ng;
//   sz=(n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng);
//   cudaMalloc(&val, sz=sizeof(T));
// }

template<typename T>
inline cudaError_t Array3D<T>::allocate_memory(integer dim1, integer dim2, integer dim3, integer n_ghost) {
  ng = n_ghost;
  n1 = dim1;
  n2 = dim2;
  n3 = dim3;
  disp2 = n3 + 2 * ng;
  disp1 = (n2 + 2 * ng) * disp2;
  dispt = (disp1 + disp2 + 1) * ng;
  sz = (n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng);
  cudaError_t err = cudaMalloc(&val, sz * sizeof(T));
  return err;
}

// template<typename T>
// inline __device__ void Array3D<T>::print_info() {
//   printf("Array info:\n Dim: (%d, %d, %d)\n", n1, n2, n3);
//   printf("ngg = %d, sz = %d\n", ng, sz);
//   //printf("Val at (5,3,0) is %e\n", this->operator()(5, 3, 0));
// }

template<typename T>
class VectorField3D {
private:
  integer ng = 0;
  integer n1 = 0, n2 = 0, n3 = 0, n4 = 0;
  integer disp1 = 0, disp2 = 0, dispt = 0;

public:
  integer sz = 0;
  T *val = nullptr;

  VectorField3D() {}

//  VectorField3D(integer dim1, integer dim2, integer dim3, integer dim4, integer n_ghost = 0) : ng(n_ghost), n1(dim1),
//                                                                                               n2(dim2), n3(dim3),
//                                                                                               n4(dim4), disp2{
//          (n3 + 2 * ng) * n4}, disp1{(n2 + 2 * ng) * disp2}, dispt{(disp1 + disp2 + n4) * ng}, sz{(n1 + 2 * ng) *
//                                                                                                  (n2 + 2 * ng) *
//                                                                                                  (n3 + 2 * ng) *
//                                                                                                  n4} { cudaMalloc(&val,
//                                                                                                                   sz *
//                                                                                                                   sizeof(T));
//  }

  cudaError_t allocate_memory(integer dim1, integer dim2, integer dim3, integer dim4, integer n_ghost = 0);

   __device__ T& operator()(const int i, const int j, const int k, const int l) {
     return val[i * disp1 + j * disp2 + k * n4 + dispt + l];
   }
   __device__ const T& operator()(const int i, const int j, const int k, const int l) const {
     return val[i * disp1 + j * disp2 + k * n4 + dispt + l];
   }

  T *data() { return val; }

  const T *data() const { return val; }

//   CUDA_CALLABLE_MEMBER ~VectorField3D() { /*cudaFree(val);*/ } // We do not deallocate the memory here because we first create a temporary VectorField on CPU with pointer to GPU memory, then we pass the handle to GPU by cudaMemcpy and the handle class is out of scope. If we deallocate the memory here, after giving out the handle, the GPU memory is also freed which is not wanted.
};

template<typename T>
inline cudaError_t
VectorField3D<T>::allocate_memory(integer dim1, integer dim2, integer dim3, integer dim4, integer n_ghost) {
  ng = n_ghost;
  n1 = dim1;
  n2 = dim2;
  n3 = dim3;
  n4 = dim4;
  disp2 = (n3 + 2 * ng) * n4;
  disp1 = (n2 + 2 * ng) * disp2;
  dispt = (disp1 + disp2 + n4) * ng;
  sz = (n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng) * n4;
  cudaError_t err = cudaMalloc(&val, sz * sizeof(T));
  return err;
}

}
#endif

namespace gxl {
/**
 * \brief An wrapper class for std::vector. A dynamic allocated array with given number of elements
 * \details Support negative index to refer to ghost elements. If used without ghost elements, then it has no difference
 * compared to std::vector
 * \tparam T type of containing elements
 */
//template<typename T>
//class Array {
//  /**
//   * \brief number of elements containing ghost elements
//   */
//  int n_elem = 0;
//  /**
//   * \brief number of ghost elements
//   */
//  int n_ghost = 0;
//  /**
//   * \brief array of elements
//   */
//  std::vector<T> val{};
//public:
//  /**
//   * \brief initialize the array with given number of elements containing ghost elements
//   * \param num_elem number of elements inside the array
//   * \param num_ghost number of ghost elements, default: 0
//   * \param init_val initial value for these elements
//   */
//  explicit Array(const int num_elem, const int num_ghost = 0, T init_val = {}) : n_elem{num_elem + 2 * num_ghost},
//                                                                                 n_ghost{num_ghost},
//                                                                                 val(n_elem, init_val) {
//  }
//
//  T &operator[](const int i) { return val[i + n_ghost]; }
//
//  const T &operator[](const int i) const { return val[i + n_ghost]; }
//
//  T &operator()(const int i) { return val[i + n_ghost]; }
//
//  const T &operator()(const int i) const { return val[i + n_ghost]; }
//
//  void resize(const int num_elem, const int num_ghost);
//};

/**
 * \brief a 3D array containing ghost elements in all directions.
 * \details for a given size mx, my, mz and a given ghost value ng, the array contains (mx+2ng)*(my+2ng)*(mz+2ng) elements
 *  and the index are allowed to be negative for the ghost elements
 * \tparam T type of containing elements
 */
template<typename T>
class Array3D {
  int n_ghost{0}, n1, n2, n3;
  int displacement_i, displacement_tot;
  std::vector<T> data_;
public:
  explicit Array3D(const int ni = 0, const int nj = 0, const int nk = 0, const int _n_ghost = 0, T dd = T{}) :
      n_ghost(_n_ghost), n1(ni), n2(nj), n3(nk), displacement_i((nj + 2 * n_ghost) * (nk + 2 * n_ghost)),
      displacement_tot((displacement_i + (nk + 2 * n_ghost) + 1) * n_ghost),
      data_((ni + 2 * n_ghost) * (nj + 2 * n_ghost) * (nk + 2 * n_ghost), dd) {
  }

  Array3D(const Array3D &arr);
  //Array3D(Array3D&& arr);

  T &operator()(const int i, const int j, const int k) {
    return data_[i * displacement_i + j * (n3 + 2 * n_ghost) + k + displacement_tot];
  }

  const T &operator()(const int i, const int j, const int k) const {
    return data_[i * displacement_i + j * (n3 + 2 * n_ghost) + k + displacement_tot];
  }

  T* data(){return data_.data();}
  const T* data() const {return data_.data();}

  auto begin() { return data_.begin(); }

  [[nodiscard]] auto cbegin() const { return data_.cbegin(); }

  auto end() { return data_.end(); }

  [[nodiscard]] auto cend() const { return data_.cend(); }

//  auto data() { return data_.data(); }

  void resize(int ni, int nj, int nk, int _n_ghost);

  void resize(int ni, int nj, int nk, int _n_ghost, T val);

  size_t size() { return data_.size(); }

  void reserve(int ni, int nj, int nk, int _n_ghost);
};

//template<typename T>
//void Array<T>::resize(const int num_elem, const int num_ghost) {
//  n_elem = num_elem + 2 * num_ghost;
//  n_ghost = num_ghost;
//  val.resize(n_elem);
//}

template<typename T>
inline Array3D<T>::Array3D(const Array3D &arr): Array3D<T>(arr.n1, arr.n2, arr.n3, arr.n_ghost) {
  data_ = arr.data_;
}

//template<typename T>
//inline Array3D<T>::Array3D(Array3D&& arr) : Array3D<T>(arr.n1, arr.n2, arr.n3, arr.n_ghost) {
//  data_ = std::move(arr.data_);
//}

template<typename DataType>
void Array3D<DataType>::resize(const int ni, const int nj, const int nk, const int _n_ghost) {
  n_ghost = _n_ghost;
  n1 = ni;
  n2 = nj;
  n3 = nk;
  data_.resize((ni + 2 * n_ghost) * (nj + 2 * n_ghost) * (nk + 2 * n_ghost));
  displacement_i = (nj + 2 * n_ghost) * (nk + 2 * n_ghost);
  displacement_tot = (displacement_i + (nk + 2 * n_ghost) + 1) * n_ghost;
}

template<typename T>
void Array3D<T>::resize(int ni, int nj, int nk, int _n_ghost, T val) {
  n_ghost = _n_ghost;
  n1 = ni + 2 * n_ghost;
  n2 = nj + 2 * n_ghost;
  n3 = nk + 2 * n_ghost;
  data_.resize(n1 * n2 * n3, val);
  displacement_i = n2 * n3;
  displacement_tot = (displacement_i + n3 + 1) * n_ghost;
  n1 = ni;
  n2 = nj;
  n3 = nk;
}

template<typename T>
void Array3D<T>::reserve(int ni, int nj, int nk, int _n_ghost) {
  const int size = (ni + 2 * _n_ghost) * (nj + 2 * _n_ghost) * (nk + 2 * _n_ghost);
  data_.reserve(size);
}

/**
 * \brief As its name, should have 2 indexes. First index supply the spatial position(in 1D) and the 2nd index is used for
 *  vector subscript. Thus, the ghost grid is only assigned for the first index.
 * \tparam T the data type of the stored datas
 */
template<typename T>
class VectorField1D {
  int n_ghost{0}, n1, n2;
  std::vector<T> data_;
  int displacement_tot;
public:
  explicit VectorField1D(const int ni = 0, const int nl = 0, const int _n_ghost = 0, T &&dd = T{}) :
      n_ghost(_n_ghost), n1(ni + 2 * n_ghost), n2(nl),
      data_(n1 * n2, std::move(dd)), displacement_tot{n2 * n_ghost} {
  }

  /**
   * \brief Get the l-th variable at position i
   * \param i x index
   * \param l variable index in a vector
   * \return the l-th variable at position i
   */
  T &operator()(const int i, const int l) {
    return data_[i * n2 + displacement_tot + l];
  }

  //Get the l-th variable at position i
  /**
   * \brief Get the constant reference of the l-th variable at position i
   * \param i x index
   * \param l variable index in a vector
   * \return the l-th variable at position i
   */
  const T &operator()(const int i, const int l) const {
    return data_[i * n2 + displacement_tot + l];
  }

  /**
   * \brief Get the pointer to the first variable at position i
   * \param i x index
i     * \return the pointer to the vector at position i
   */
  auto operator()(const int i) {
    return data_.data() + i * n2 + displacement_tot;
  }

  /**
   * \brief return the number of variables of the vector field
   * \return number of variables/dimension of the vector
   */
  [[nodiscard]] int n_variable() const { return n2; }

  void resize(int nx, int nl, int ngg);
};

/**
 * \brief As its name, should have 4 indexes. First 3 index supply the spatial position and the 4th index is used for
 *  vector subscript. Thus, the ghost grid is only assigned for first 3 index.
 * \tparam T the data type of the stored datas
 */
template<typename T>
class VectorField3D {
  int n_ghost{0}, n1, n2, n3, n4;
  std::vector<T> data_;
  int displacement_j, displacement_i, displacement_tot;
public:
  explicit VectorField3D(const int ni = 0, const int nj = 0, const int nk = 0, const int nl = 0, const int _n_ghost = 0,
                         T &&dd = T{}) :
      n_ghost(_n_ghost), n1(ni), n2(nj), n3(nk), n4(nl),
      data_((ni + 2 * n_ghost) * (nj + 2 * n_ghost) * (nk + 2 * n_ghost) * n4, std::move(dd)),
      displacement_j{(nk + 2 * n_ghost) * n4}, displacement_i((nj + 2 * n_ghost) * displacement_j),
      displacement_tot((displacement_i + displacement_j + n4) * n_ghost) {
  }

  auto data() { return data_.data(); }

  /**
   * \brief Get the l-th variable at position (i,j,k)
   * \param i x index
   * \param j y index
   * \param k z index
   * \param l variable index in a vector
   * \return the l-th variable at position (i,j,k)
   */
  T &operator()(const int i, const int j, const int k, const int l) {
    return data_[i * displacement_i + j * displacement_j + k * n4 + displacement_tot + l];
  }

  //Get the l-th variable at position (i,j,k)
  /**
   * \brief Get the constant reference of the l-th variable at position (i,j,k)
   * \param i x index
   * \param j y index
   * \param k z index
   * \param l variable index in a vector
   * \return the l-th variable at position (i,j,k)
   */
  const T &operator()(const int i, const int j, const int k, const int l) const {
    return data_[i * displacement_i + j * displacement_j + k * n4 + displacement_tot + l];
  }

  /**
   * \brief Get the pointer to the first variable at position (i,j,k)
   * \param i x index
   * \param j y index
   * \param k z index
   * \return the pointer to the vector at position (i,j,k)
   */
  auto operator()(const int i, const int j, const int k) {
    return data_.data() + i * displacement_i + j * displacement_j + k * n4 + displacement_tot;
  }

  /**
 * \brief Get the pointer to the first variable at position (i,j,k)
 * \param i x index
 * \param j y index
 * \param k z index
 * \return the pointer to the vector at position (i,j,k)
 */
  auto operator()(const int i, const int j, const int k) const {
    return data_.data() + i * displacement_i + j * displacement_j + k * n4 + displacement_tot;
  }

  /**
   * \brief return the number of variables of the vector field
   * \return number of variables/dimension of the vector
   */
  [[nodiscard]] int n_variable() const { return n4; }

  void reserve(int ni, int nj, int nk, int nl, int ngg);

  void resize(int ni, int nj, int nk, int nl, int ngg);
};

template<typename T>
void VectorField1D<T>::resize(int nx, int nl, int ngg) {
  n_ghost = ngg;
  n1 = nx + 2 * ngg;
  n2 = nl;
  data_.resize(n1 * n2);
  displacement_tot = n2 * ngg;
}

template<typename T>
void VectorField3D<T>::reserve(const int ni, const int nj, const int nk, const int nl, const int ngg) {
  const int size = (ni + 2 * ngg) * (nj + 2 * ngg) * (nk + 2 * ngg) * nl;
  data_.reserve(size);
}

template<typename T>
void VectorField3D<T>::resize(int ni, int nj, int nk, int nl, int ngg) {
  n_ghost = ngg;
  n1 = ni + 2 * ngg;
  n2 = nj + 2 * ngg;
  n3 = nk + 2 * ngg;
  n4 = nl;
  displacement_j = n3 * n4;
  displacement_i = n2 * displacement_j;
  displacement_tot = (displacement_i + displacement_j + n4) * n_ghost;
  data_.resize(n1 * n2 * n3 * n4);
  n1 = ni;
  n2 = nj;
  n3 = nk;
}

}
