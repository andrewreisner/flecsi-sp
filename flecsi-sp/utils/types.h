
#include <flecsi/data/common/privilege.h>
#include <flecsi/data/data_client_handle.h>
#include <flecsi/data/dense_accessor.h>
#include <flecsi/data/sparse_accessor.h>
#include <flecsi/data/sparse_mutator.h>

#pragma once

namespace flecsi_sp {
namespace utils {

// the dense handle type
template<typename T>
using dense_handle_w =
  flecsi::dense_accessor<T, flecsi::wo, flecsi::wo, flecsi::na>;

template<typename T>
using dense_handle_rw =
  flecsi::dense_accessor<T, flecsi::rw, flecsi::rw, flecsi::ro>;

template<typename T>
using dense_handle_r =
  flecsi::dense_accessor<T, flecsi::ro, flecsi::ro, flecsi::ro>;

// the sparse handle type
template<typename T>
using sparse_mutator =
  flecsi::sparse_mutator<T>;

template<typename T>
using sparse_handle_w =
  flecsi::sparse_accessor<T, flecsi::wo, flecsi::wo, flecsi::ro>;

template<typename T>
using sparse_handle_rw =
  flecsi::sparse_accessor<T, flecsi::rw, flecsi::rw, flecsi::ro>;

template<typename T>
using sparse_handle_r =
  flecsi::sparse_accessor<T, flecsi::ro, flecsi::ro, flecsi::ro>;


// the client handles
template<typename DC>
using client_handle_w = flecsi::data_client_handle_u<DC, flecsi::wo>;

template<typename DC>
using client_handle_r = flecsi::data_client_handle_u<DC, flecsi::ro>;

template<typename DC>
using client_handle_rw = flecsi::data_client_handle_u<DC, flecsi::rw>;

} // namespace
} // namespace
