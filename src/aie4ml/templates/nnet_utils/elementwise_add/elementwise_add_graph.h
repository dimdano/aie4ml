#pragma once
#include <adf.h>
#include "elementwise_add.h"
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class elementwise_add_graph : public graph {
public:
  static constexpr unsigned CAS_NUM = ConfigT::CAS_NUM;
  static constexpr unsigned TILE_ELEMENTS = ConfigT::TILE_ELEMENTS;

  input_port in1[CAS_NUM];
  input_port in2[CAS_NUM];
  output_port out1[CAS_NUM];
  kernel kk[CAS_NUM];

  void place_graph(int COL_START, int ROW_START)
  {
    for (int row = 0; row < CAS_NUM; ++row)
    {
      const int tileRow = ROW_START + row;
      const int tileCol = COL_START;
      adf::location<adf::kernel>(kk[row]) = adf::tile(tileCol, tileRow);
      adf::location<adf::buffer>(kk[row].in[0]) = {
        adf::bank(tileCol - 1, tileRow, 0),
        adf::bank(tileCol - 1, tileRow, 3)
      };
      adf::location<adf::stack>(kk[row]) = adf::bank(tileCol - 1, tileRow, 1);
      adf::location<adf::buffer>(kk[row].in[1]) = {
        adf::bank(tileCol, tileRow, 1),
        adf::bank(tileCol, tileRow, 2)
      };
      adf::location<adf::buffer>(kk[row].out[0]) = {
        adf::bank(tileCol, tileRow, 0),
        adf::bank(tileCol, tileRow, 3)
      };
    }
  }

  elementwise_add_graph()
  {
    for (int row = 0; row < CAS_NUM; ++row) {
      kk[row] = kernel::create_object<elementwise_add_kernel<ConfigT>>();
      source(kk[row]) = "elementwise_add.cpp";
      runtime<ratio>(kk[row]) = 1.0;
    }

    for (int row = 0; row < CAS_NUM; ++row) {
      connect<>(in1[row], kk[row].in[0]);
      connect<>(in2[row], kk[row].in[1]);
      connect<>(kk[row].out[0], out1[row]);
      dimensions(kk[row].in[0]) = {TILE_ELEMENTS};
      dimensions(kk[row].in[1]) = {TILE_ELEMENTS};
      dimensions(kk[row].out[0]) = {TILE_ELEMENTS};
    }
  }
};
