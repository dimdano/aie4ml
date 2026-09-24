// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <adf.h>
#include <utility>
#include "conv2d_graph.h"
#include "frame_retile.h"
#include "parameters.h"

using namespace adf;

// A retiler kernel per band, band `b` on row `b` beside the conv tile it feeds, on the Dense bank
// contract: its input and frame where IN1/OUT1_BUFFER_LOCATIONS put them -- one copy per bank -- and
// its stack in bank 1. Each band's kernel is configured by `ConfigT::band<b>::type`.
template<typename ConfigT>
class frame_retile_graph : public graph {
public:
  static constexpr unsigned BANDS = ConfigT::BANDS;
  input_port in1[BANDS];
  output_port out1[BANDS];
  kernel kk[BANDS];

  void place_graph(int COL_START, int ROW_START)
  {
    for (unsigned band = 0; band < BANDS; ++band) {
      adf::location<adf::kernel>(kk[band]) = adf::tile(COL_START, ROW_START + band);
      adf::location<adf::stack>(kk[band]) = adf::bank(COL_START, ROW_START + band, 1);
      conv2d_pin_buffer(kk[band].in[0], ConfigT::IN1_BUFFER_LOCATIONS[band], COL_START, ROW_START);
      conv2d_pin_buffer(kk[band].out[0], ConfigT::OUT1_BUFFER_LOCATIONS[band], COL_START, ROW_START);
    }
  }

  frame_retile_graph( void ) { build(std::make_integer_sequence<unsigned, BANDS>{}); }

private:
  template<unsigned... BAND>
  void build(std::integer_sequence<unsigned, BAND...>) { (build_band<BAND>(), ...); }

  template<unsigned BAND>
  void build_band()
  {
    using BandT = typename ConfigT::template band<BAND>::type;
    kk[BAND] = kernel::create_object<frame_retile<BandT>>();
    source(kk[BAND]) = "frame_retile.cpp";
    runtime<ratio>(kk[BAND]) = 1.0;
    dimensions(kk[BAND].in[0]) = { BandT::SRC_BYTES };
    dimensions(kk[BAND].out[0]) = { ConfigT::FRAME_BYTES };
    connect<>(in1[BAND], kk[BAND].in[0]);
    connect<>(kk[BAND].out[0], out1[BAND]);
  }
};
