#pragma once
#ifdef __CUDACC__

#include <cuda_runtime.h>

#endif

#include <vector>

enum class Major {
  ColMajor = 0,
  RowMajor
};

#ifdef __CUDACC__
namespace ggxl {
template<typename T, Major major = Major::ColMajor>
class Array3D {
private:
  int disp1 = 0, disp2 = 0, dispt = 0;
  int sz = 0;
  T *val = nullptr;
  int ng = 0;
  int n1 = 0, n2 = 0, n3 = 0;

public:
//  Array3D() {}

  // Shallow copy constructor
//  Array3D(const Array3D &b) : ng(b.ng), n1(b.n1), n2(b.n2), n3(b.n3), disp1(b.disp1), disp2(b.disp2), dispt(b.dispt),
//                              sz(b.sz), val(b.val) {}

  cudaError_t allocate_memory(int dim1, int dim2, int dim3, int n_ghost = 0);

  __device__ T &operator()(const int i, const int j, const int k) {
    if constexpr (major == Major::ColMajor)
      return val[k * disp1 + j * disp2 + i + dispt];
    else
      return val[i * disp1 + j * disp2 + k + dispt];
  }

  __device__ const T &operator()(const int i, const int j, const int k) const {
    if constexpr (major == Major::ColMajor) {
      return val[k * disp1 + j * disp2 + i + dispt];
    } else {
      return val[i * disp1 + j * disp2 + k + dispt];
    }
  }

  T *data() { return val; }

//  const T *data() const { return val; }

  auto size() { return sz; }
};

template<typename T, Major major>
inline cudaError_t Array3D<T, major>::allocate_memory(int dim1, int dim2, int dim3, int n_ghost) {
  ng = n_ghost;
  n1 = dim1;
  n2 = dim2;
  n3 = dim3;
  if constexpr (major == Major::ColMajor) {
    disp2 = n1 + 2 * ng;
  } else {
    disp2 = n3 + 2 * ng;
  }
  disp1 = (n2 + 2 * ng) * disp2;
  dispt = (disp1 + disp2 + 1) * ng;
  sz = (n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng);
  cudaError_t err = cudaMalloc(&val, sz * sizeof(T));
  return err;
}

template<typename T, Major major = Major::ColMajor>
class VectorField3D {
private:
  int disp1 = 0, disp2 = 0, dispt = 0;
  int sz = 0;
  T *val = nullptr;
  int ng = 0;
  int n1 = 0, n2 = 0, n3 = 0, n4 = 0;

public:

//  VectorField3D() {}

  cudaError_t allocate_memory(int dim1, int dim2, int dim3, int dim4, int n_ghost = 0);

  __device__ T &operator()(const int i, const int j, const int k, const int l) {
    if constexpr (major == Major::RowMajor) {
      return val[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return val[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
  }

  __device__ const T &operator()(const int i, const int j, const int k, const int l) const {
    if constexpr (major == Major::RowMajor) {
      return val[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return val[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
  }

  T *data() { return val; }

  auto size() { return sz; }
//  const T *data() const { return val; }

//   CUDA_CALLABLE_MEMBER ~VectorField3D() { /*cudaFree(val);*/ } // We do not deallocate the memory here because we first create a temporary VectorField on CPU with pointer to GPU memory, then we pass the handle to GPU by cudaMemcpy and the handle class is out of scope. If we deallocate the memory here, after giving out the handle, the GPU memory is also freed which is not wanted.
};

template<typename T, Major major>
inline cudaError_t
VectorField3D<T, major>::allocate_memory(int dim1, int dim2, int dim3, int dim4, int n_ghost) {
  ng = n_ghost;
  n1 = dim1;
  n2 = dim2;
  n3 = dim3;
  n4 = dim4;
  if constexpr (major == Major::RowMajor) {
    disp2 = (n3 + 2 * ng) * n4;
    disp1 = (n2 + 2 * ng) * disp2;
    dispt = (disp1 + disp2 + n4) * ng;
  } else { // Column major
    disp2 = n1 + 2 * ng;
    disp1 = (n2 + 2 * ng) * disp2;
    dispt = (disp1 + disp2 + 1) * ng;
  }
  sz = (n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng);
  cudaError_t err = cudaMalloc(&val, sz * n4 * sizeof(T));
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
//  int ng = 0;
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
//                                                                                 ng{num_ghost},
//                                                                                 val(n_elem, init_val) {
//  }
//
//  T &operator[](const int i) { return val[i + ng]; }
//
//  const T &operator[](const int i) const { return val[i + ng]; }
//
//  T &operator()(const int i) { return val[i + ng]; }
//
//  const T &operator()(const int i) const { return val[i + ng]; }
//
//  void resize(const int num_elem, const int num_ghost);
//};

/**
 * \brief a 3D array containing ghost elements in all directions.
 * \details for a given size mx, my, mz and a given ghost value ng, the array contains (mx+2ng)*(my+2ng)*(mz+2ng) elements
 *  and the index are allowed to be negative for the ghost elements
 * \tparam T type of containing elements
 */
template<typename T, Major major = Major::ColMajor>
class Array3D {
  int ng{0}, n1, n2, n3;
  int disp2, disp1, dispt;
  std::vector<T> data_;
public:
  explicit Array3D(const int ni = 0, const int nj = 0, const int nk = 0, const int _n_ghost = 0, T dd = T{}) :
      ng(_n_ghost), n1(ni), n2(nj), n3(nk), disp2(n1 + 2 * ng), disp1((n2 + 2 * ng) * disp2),
      dispt((disp1 + disp2 + 1) * ng),
      data_((ni + 2 * ng) * (nj + 2 * ng) * (nk + 2 * ng), dd) {
    if constexpr (major == Major::RowMajor) {
      disp2 = n3 + 2 * ng;
      disp1 = (n2 + 2 * ng) * disp2;
      dispt = (disp1 + disp2 + 1) * ng;
    }
  }

  Array3D(const Array3D &arr);

  T &operator()(const int i, const int j, const int k) {
    if constexpr (major == Major::RowMajor) {
      return data_[i * disp1 + j * disp2 + k + dispt];
    } else {
      return data_[k * disp1 + j * disp2 + i + dispt];
    }
  }

  const T &operator()(const int i, const int j, const int k) const {
    if constexpr (major == Major::RowMajor) {
      return data_[i * disp1 + j * disp2 + k + dispt];
    } else {
      return data_[k * disp1 + j * disp2 + i + dispt];
    }
  }

  T *data() { return data_.data(); }

  const T *data() const { return data_.data(); }

  auto begin() { return data_.begin(); }

//  [[nodiscard]] auto cbegin() const { return data_.cbegin(); }

  auto end() { return data_.end(); }

//  [[nodiscard]] auto cend() const { return data_.cend(); }

  void resize(int ni, int nj, int nk, int _n_ghost, T dd = T{});

//  void resize(int ni, int nj, int nk, int _n_ghost, T val);
//
//  size_t size() { return data_.size(); }
//
//  void reserve(int ni, int nj, int nk, int _n_ghost);
};

template<typename T, Major major>
inline Array3D<T, major>::Array3D(const Array3D &arr): Array3D<T, major>(arr.n1, arr.n2, arr.n3, arr.ng) {
  data_ = arr.data_;
}

template<typename DataType, Major major>
void Array3D<DataType, major>::resize(const int ni, const int nj, const int nk, const int _n_ghost, DataType dd) {
  ng = _n_ghost;
  n1 = ni;
  n2 = nj;
  n3 = nk;
  data_.resize((ni + 2 * ng) * (nj + 2 * ng) * (nk + 2 * ng), dd);
  if constexpr (major == Major::RowMajor) {
    disp2 = n3 + 2 * ng;
  } else {
    disp2 = n1 + 2 * ng;
  }
  disp1 = (n2 + 2 * ng) * disp2;
  dispt = (disp1 + disp2 + 1) * ng;
}

//template<typename T, Major major>
//void Array3D<T, major>::resize(int ni, int nj, int nk, int _n_ghost, T val) {
//  ng = _n_ghost;
//  n1 = ni + 2 * ng;
//  n2 = nj + 2 * ng;
//  n3 = nk + 2 * ng;
//  data_.resize(n1 * n2 * n3, val);
//  if constexpr (major == Major::RowMajor) {
//    disp2 = n3;
//  } else {
//    disp2 = n1;
//  }
//  disp1 = n2 * disp2;
//  dispt = (disp1 + disp2 + 1) * ng;
//  n1 = ni;
//  n2 = nj;
//  n3 = nk;
//}
//
//template<typename T, Major major>
//void Array3D<T, major>::reserve(int ni, int nj, int nk, int _n_ghost) {
//  const int size = (ni + 2 * _n_ghost) * (nj + 2 * _n_ghost) * (nk + 2 * _n_ghost);
//  data_.reserve(size);
//}

/**
 * \brief As its name, should have 2 indexes. First index supply the spatial position(in 1D) and the 2nd index is used for
 *  vector subscript. Thus, the ghost grid is only assigned for the first index.
 * \tparam T the data type of the stored datas
 */
//template<typename T>
//class VectorField1D {
//  int ng{0}, n1, n2;
//  std::vector<T> data_;
//  int dispt;
//public:
//  explicit VectorField1D(const int ni = 0, const int nl = 0, const int _n_ghost = 0, T &&dd = T{}) :
//      ng(_n_ghost), n1(ni + 2 * ng), n2(nl),
//      data_(n1 * n2, std::move(dd)), dispt{n2 * ng} {
//  }
//
//  /**
//   * \brief Get the l-th variable at position i
//   * \param i x index
//   * \param l variable index in a vector
//   * \return the l-th variable at position i
//   */
//  T &operator()(const int i, const int l) {
//    return data_[i * n2 + dispt + l];
//  }
//
//  //Get the l-th variable at position i
//  /**
//   * \brief Get the constant reference of the l-th variable at position i
//   * \param i x index
//   * \param l variable index in a vector
//   * \return the l-th variable at position i
//   */
//  const T &operator()(const int i, const int l) const {
//    return data_[i * n2 + dispt + l];
//  }
//
//  /**
//   * \brief Get the pointer to the first variable at position i
//   * \param i x index
//i     * \return the pointer to the vector at position i
//   */
//  auto operator()(const int i) {
//    return data_.data() + i * n2 + dispt;
//  }
//
//  /**
//   * \brief return the number of variables of the vector field
//   * \return number of variables/dimension of the vector
//   */
//  [[nodiscard]] int n_variable() const { return n2; }
//
//  void resize(int nx, int nl, int ngg);
//};

/**
 * \brief As its name, should have 4 indexes. First 3 index supply the spatial position and the 4th index is used for
 *  vector subscript. Thus, the ghost grid is only assigned for first 3 index.
 * \tparam T the data type of the stored datas
 */
template<typename T, Major major = Major::ColMajor>
class VectorField3D {
  int ng{0}, n1, n2, n3, n4, sz;
  std::vector<T> data_;
  int disp2, disp1, dispt;
public:
//  explicit VectorField3D(const int ni = 0, const int nj = 0, const int nk = 0, const int nl = 0, const int _n_ghost = 0,
//                         T &&dd = T{}) :
//      ng(_n_ghost), n1(ni), n2(nj), n3(nk), n4(nl), sz((ni + 2 * ng) * (nj + 2 * ng) * (nk + 2 * ng)),
//      data_(sz * n4, std::move(dd)),
//      disp2{n1 + 2 * ng}, disp1((n2 + 2 * ng) * disp2),
//      dispt((disp1 + disp2 + 1) * ng) {
//    if constexpr (major == Major::RowMajor) {
//      disp2 = (n3 + 2 * ng) * n4;
//      disp1 = (n2 + 2 * ng) * disp2;
//      dispt = (disp1 + disp2 + n4) * ng;
//    }
//  }

  auto data() const { return data_.data(); }

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
    if constexpr (major == Major::RowMajor) {
      return data_[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return data_[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
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
//  const T &operator()(const int i, const int j, const int k, const int l) const {
//    if constexpr (major == Major::RowMajor) {
//      return data_[i * disp1 + j * disp2 + k * n4 + dispt + l];
//    } else {
//      return data_[k * disp1 + j * disp2 + i + dispt + l * sz];
//    }
//  }

  /**
   * \brief Get the pointer to the first variable at position (i,j,k)
   * \param i x index
   * \param j y index
   * \param k z index
   * \return the pointer to the vector at position (i,j,k)
   */
//  auto operator()(const int i, const int j, const int k) {
//    return data_.data() + i * disp1 + j * disp2 + k * n4 + dispt;
//  }

  /**
 * \brief Get the pointer to the first variable at position (i,j,k)
 * \param i x index
 * \param j y index
 * \param k z index
 * \return the pointer to the vector at position (i,j,k)
 */
//  auto operator()(const int i, const int j, const int k) const {
//    return data_.data() + i * disp1 + j * disp2 + k * n4 + dispt;
//  }

  /**
   * \brief return the number of variables of the vector field
   * \return number of variables/dimension of the vector
   */
//  [[nodiscard]] int n_variable() const { return n4; }
//
//  void reserve(int ni, int nj, int nk, int nl, int ngg);

  void resize(int ni, int nj, int nk, int nl, int ngg, T&& t=T{});

//  auto size() { return data_.size(); }
};

//template<typename T>
//void VectorField1D<T>::resize(int nx, int nl, int ngg) {
//  ng = ngg;
//  n1 = nx + 2 * ngg;
//  n2 = nl;
//  data_.resize(n1 * n2);
//  dispt = n2 * ngg;
//}

//template<typename T, Major major>
//void VectorField3D<T, major>::reserve(const int ni, const int nj, const int nk, const int nl, const int ngg) {
//  const int size = (ni + 2 * ngg) * (nj + 2 * ngg) * (nk + 2 * ngg) * nl;
//  data_.reserve(size);
//}

template<typename T, Major major>
void VectorField3D<T, major>::resize(int ni, int nj, int nk, int nl, int ngg, T&& t) {
  ng = ngg;
  n1 = ni + 2 * ngg;
  n2 = nj + 2 * ngg;
  n3 = nk + 2 * ngg;
  n4 = nl;
  sz = n1 * n2 * n3;
  if constexpr (major == Major::RowMajor) {
    disp2 = n3 * n4;
    disp1 = n2 * disp2;
    dispt = (disp1 + disp2 + n4) * ng;
  } else {
    disp2 = n1;
    disp1 = n2 * disp2;
    dispt = (disp1 + disp2 + 1) * ng;
  }
  data_.resize(n1 * n2 * n3 * n4, t);
  n1 = ni;
  n2 = nj;
  n3 = nk;
}

}
