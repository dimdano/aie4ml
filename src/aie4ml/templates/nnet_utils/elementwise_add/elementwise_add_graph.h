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
      const auto lhsLocation = ConfigT::IN1_BUFFER_LOCATIONS[row];
      const auto rhsLocation = ConfigT::IN2_BUFFER_LOCATIONS[row];
      const auto outputLocation = ConfigT::OUT1_BUFFER_LOCATIONS[row];
      const int tileCol = COL_START;
      const int tileRow = ROW_START + row;
      adf::location<adf::kernel>(kk[row]) = adf::tile(tileCol, tileRow);
      if (lhsLocation.bank_count == 1) {
        adf::location<adf::buffer>(kk[row].in[0]) = adf::bank(
          COL_START + lhsLocation.col, ROW_START + lhsLocation.row, lhsLocation.bank0);
      } else {
        adf::location<adf::buffer>(kk[row].in[0]) = {
          adf::bank(COL_START + lhsLocation.col, ROW_START + lhsLocation.row, lhsLocation.bank0),
          adf::bank(COL_START + lhsLocation.col, ROW_START + lhsLocation.row, lhsLocation.bank1)
        };
      }
      adf::location<adf::stack>(kk[row]) = adf::bank(tileCol, tileRow, 1);
      if (rhsLocation.bank_count == 1) {
        adf::location<adf::buffer>(kk[row].in[1]) = adf::bank(
          COL_START + rhsLocation.col, ROW_START + rhsLocation.row, rhsLocation.bank0);
      } else {
        adf::location<adf::buffer>(kk[row].in[1]) = {
          adf::bank(COL_START + rhsLocation.col, ROW_START + rhsLocation.row, rhsLocation.bank0),
          adf::bank(COL_START + rhsLocation.col, ROW_START + rhsLocation.row, rhsLocation.bank1)
        };
      }
      if (outputLocation.bank_count == 1) {
        adf::location<adf::buffer>(kk[row].out[0]) = adf::bank(
          COL_START + outputLocation.col, ROW_START + outputLocation.row, outputLocation.bank0);
      } else {
        adf::location<adf::buffer>(kk[row].out[0]) = {
          adf::bank(COL_START + outputLocation.col, ROW_START + outputLocation.row, outputLocation.bank0),
          adf::bank(COL_START + outputLocation.col, ROW_START + outputLocation.row, outputLocation.bank1)
        };
      }
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
