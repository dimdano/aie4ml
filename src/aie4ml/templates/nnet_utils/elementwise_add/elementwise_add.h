#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class elementwise_add_base {
public:
  using lhs_t = typename ConfigT::lhs_t;
  using rhs_t = typename ConfigT::rhs_t;
  using result_t = typename ConfigT::result_t;
  using acc_scalar_t = typename ConfigT::acc_scalar_t;

  elementwise_add_base();
};

template<typename ConfigT>
class elementwise_add_kernel : public elementwise_add_base<ConfigT> {
public:
  using lhs_t = typename ConfigT::lhs_t;
  using rhs_t = typename ConfigT::rhs_t;
  using result_t = typename ConfigT::result_t;
  using acc_scalar_t = typename elementwise_add_base<ConfigT>::acc_scalar_t;

  void run(input_buffer<lhs_t>& lhs, input_buffer<rhs_t>& rhs, output_buffer<result_t>& out);

  static void registerKernelClass() {
    REGISTER_FUNCTION(elementwise_add_kernel::run);
  }
};
