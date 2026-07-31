// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <adf.h>
#include "softmax.h"
#include "parameters.h"

using namespace adf;

template <typename ConfigT>
class softmax_hccs_graph : public graph {
public:
    static constexpr int CAS_NUM = ConfigT::CAS_NUM;
    static constexpr int ROWS    = ConfigT::ROWS;
    static constexpr int COLS    = ConfigT::COLS;

    input_port  in1[CAS_NUM];
    output_port out1[CAS_NUM];

    kernel kk[CAS_NUM];

    void place_graph(int COL_START, int ROW_START)
    {
        for (int row = 0; row < CAS_NUM; ++row) {
            const int tileRow = ROW_START + row;
            const int tileCol = COL_START;

            adf::location<adf::kernel>(kk[row]) = adf::tile(tileCol, tileRow);

            adf::location<adf::buffer>(kk[row].in[0]) = {
                adf::bank(tileCol - 1, tileRow, 0),
                adf::bank(tileCol - 1, tileRow, 3)
            };
            adf::location<adf::stack>(kk[row]) = adf::bank(tileCol - 1, tileRow, 1);

            adf::location<adf::buffer>(kk[row].out[0]) = {
                adf::bank(tileCol, tileRow, 0),
                adf::bank(tileCol, tileRow, 3)
            };
        }
    }

    softmax_hccs_graph()
    {
        for (int i = 0; i < CAS_NUM; ++i) {
            if constexpr (ConfigT::APPROX_EXP) {
                if constexpr (ConfigT::LAYOUT_TILED) {
                    kk[i] = kernel::create_object<softmax_exp_i8_tiled<ConfigT>>();
                } else {
                    kk[i] = kernel::create_object<softmax_exp_i8<ConfigT>>();
                }
            } else if constexpr (ConfigT::LAYOUT_TILED) {
                kk[i] = kernel::create_object<softmax_i8_tiled<ConfigT>>(
                    ConfigT::B[i], ConfigT::S[i], ConfigT::Dmax[i]);
            } else {
                kk[i] = kernel::create_object<softmax_i8<ConfigT>>(
                    ConfigT::B[i], ConfigT::S[i], ConfigT::Dmax[i]);
            }
            source(kk[i])         = "softmax.cpp";
            runtime<ratio>(kk[i]) = 1.0;

            connect<>(in1[i], kk[i].in[0]);
            dimensions(kk[i].in[0])  = {ROWS * COLS};
            dimensions(kk[i].out[0]) = {ROWS * COLS};
            connect<>(kk[i].out[0], out1[i]);
        }
    }
};
