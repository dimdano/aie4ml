// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <adf.h>
#include <vector>
#include "dense_bias_relu.h"
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class dense_bias_relu_graph : public graph {
public:
  static constexpr unsigned IN_FEAT  = ConfigT::IN_FEAT;
  static constexpr unsigned OUT_FEAT = ConfigT::OUT_FEAT;
  static constexpr unsigned CAS_NUM  = ConfigT::CAS_NUM;
  static constexpr unsigned CAS_LENGTH = ConfigT::CAS_LENGTH;
  static constexpr unsigned IN_FEAT_SLICE  = ConfigT::IN_FEAT_SLICE;
  static constexpr unsigned OUT_FEAT_SLICE = ConfigT::OUT_FEAT_SLICE;
  static constexpr unsigned padded_independent_extent = ConfigT::padded_independent_extent;
  static constexpr int M  = ConfigT::M;
  static constexpr int K  = ConfigT::K;
  static constexpr int N  = ConfigT::N;

  input_port  in1[CAS_LENGTH];
  adf::port<adf::direction::in> wts[CAS_NUM * CAS_LENGTH];
  adf::port<adf::direction::in> bias[CAS_NUM];
  output_port out1[CAS_NUM];
  kernel kk[CAS_NUM * CAS_LENGTH]; // row wise

  void place_graph(int COL_START, int ROW_START)
  {
    // NOTE this placement works only for BufferOptLevel8
    // TODO find why the rtp buffers need one additional bank
    // TODO examine if co-locating this bank with another bank introduces memory delays
    for (int idx = 0; idx < CAS_NUM * CAS_LENGTH; ++idx)
    {
      int tileRow = ROW_START + (idx / CAS_LENGTH);
      int tileCol = COL_START + (idx % CAS_LENGTH);
      adf::location<adf::kernel>(kk[idx]) = adf::tile(tileCol, tileRow);
      // place buffers on west tile because each AI Engine-ML accesses its own memory to the east
      adf::location<adf::buffer>(kk[idx].in[0]) = { adf::bank(tileCol-1, tileRow, 0), adf::bank(tileCol-1, tileRow, 3) };
      adf::location<adf::buffer>(kk[idx].in[1]) = adf::bank(tileCol-1, tileRow, 1);
      adf::location<adf::stack>(kk[idx]) = adf::bank(tileCol-1, tileRow, 2);
      if (idx % CAS_LENGTH == CAS_LENGTH - 1){
          adf::location<adf::buffer>(kk[idx].out[0]) = { adf::bank(tileCol, tileRow, 1), adf::bank(tileCol, tileRow, 2) };
          if (CAS_LENGTH == 1){ // match the bias argument for kernel dense_first/dense_last
              adf::location<adf::buffer>(kk[idx].in[2]) = adf::bank(tileCol, tileRow, 0);
          }else{
              adf::location<adf::buffer>(kk[idx].in[3]) = adf::bank(tileCol, tileRow, 0);
          }
      }
    }
  }

  dense_bias_relu_graph( void )
  {

    for (int chain = 0; chain < CAS_NUM; ++chain) {
        if constexpr (CAS_LENGTH == 1) {
            kk[chain * CAS_LENGTH + 0] = kernel::create_object<dense_single<ConfigT>>();
        }
        else {
            kk[chain * CAS_LENGTH + 0] = kernel::create_object<dense_first<ConfigT>>();
            if constexpr (CAS_LENGTH > 2) {
                for (int c = 1; c < CAS_LENGTH - 1; ++c) {
                    kk[chain * CAS_LENGTH + c] = kernel::create_object<dense_middle<ConfigT>>();
                }
            }
            kk[chain * CAS_LENGTH + (CAS_LENGTH - 1)] = kernel::create_object<dense_last<ConfigT>>();
        }
    }

    for (int idx = 0; idx < CAS_LENGTH * CAS_NUM; ++idx) {
        int col = idx % CAS_LENGTH;
        int row = idx / CAS_LENGTH;
        source(kk[idx])        = "dense_bias_relu.cpp";
        runtime<ratio>(kk[idx]) = 1.0;
        single_buffer(kk[idx].in[1]);
        connect<parameter>(wts[idx], async(kk[idx].in[1]));
        if (col == CAS_LENGTH - 1){
          if (col == 0){ // match the bias argument for kernel dense_first/dense_last
            connect<parameter>(bias[row], async(kk[idx].in[2]));
            single_buffer(kk[idx].in[2]);
           } else {
            connect<parameter>(bias[row], async(kk[idx].in[3]));
            single_buffer(kk[idx].in[3]);
           }
        }

    }

    for (unsigned col = 0; col < CAS_LENGTH; ++col) {
      for (unsigned ch = 0; ch < CAS_NUM; ++ch) {
        int idx = ch*CAS_LENGTH + col;
        connect<>( in1[col], kk[idx].in[0] );
        dimensions( kk[idx].in[0] ) = { padded_independent_extent * IN_FEAT_SLICE };
      }
    }

    for (int chain = 0; chain < CAS_NUM; ++chain) {
        const int last_idx = chain * CAS_LENGTH + (CAS_LENGTH - 1);
        connect<>( kk[last_idx].out[0], out1[chain] );
        dimensions( kk[last_idx].out[0] ) = { padded_independent_extent * OUT_FEAT_SLICE };
    }

    if constexpr (CAS_LENGTH > 1) {
      for (int chain = 0; chain < CAS_NUM; ++chain) {
        for (int c = 0; c < CAS_LENGTH - 1; ++c) {
          connect<cascade>(
            kk[chain * CAS_LENGTH + c].out[0],
            kk[chain * CAS_LENGTH + c + 1].in[2]
          );
        }
      }
    }

  }

};
