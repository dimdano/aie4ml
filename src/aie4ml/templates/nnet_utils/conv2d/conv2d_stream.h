// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Stream-port flavour of the Conv2D kernel (ConfigT::STREAM_IO). The frame arrives and the result
// leaves on core streams in linear row order -- rows, then columns, then channels -- which is the
// wire order every stream port in aie4ml carries. The kernel lands that wire into the
// channel-blocked frame the shared compute core reads, and unpacks the result on the way out.

#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class conv2d_stream {
public:
  using data_t   = typename ConfigT::data_t;
  using weight_t = typename ConfigT::weight_t;
  using result_t = typename ConfigT::result_t;
  using bias_t   = typename ConfigT::bias_t;

  conv2d_stream();

  void run(input_stream<data_t>*  ifm,
           const weight_t (&wts)[ConfigT::WN],
           const bias_t (&bias)[ConfigT::BN],
           output_stream<result_t>* ofm);

  static void registerKernelClass() { REGISTER_FUNCTION(conv2d_stream::run); }
};
