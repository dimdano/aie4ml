#pragma once
#include <adf.h>
#include <cstdint>
#include "matmul.h"
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class matmul_graph : public graph {
public:
  static constexpr unsigned CAS_NUM    = ConfigT::CAS_NUM;
  static constexpr unsigned CAS_LENGTH = ConfigT::CAS_LENGTH;
  static constexpr unsigned padded_M   = ConfigT::padded_M;
  static constexpr unsigned K_SLICE    = ConfigT::K_SLICE;
  static constexpr unsigned N_SLICE    = ConfigT::N_SLICE;

  // 'outer' splits the rows across the chains, so each tile reads its own lhs slice rather
  // than sharing one per cascade column. padded_M is already the per-tile row extent.
  static constexpr bool PARALLELISM_CONTRACT_OUTER = ConfigT::PARALLELISM_CONTRACT_OUTER;
  static constexpr unsigned LHS_PORTS = PARALLELISM_CONTRACT_OUTER ? CAS_NUM * CAS_LENGTH : CAS_LENGTH;

  using a_t = typename ConfigT::a_t;
  using b_t = typename ConfigT::b_t;
  using c_t = typename ConfigT::c_t;

  static constexpr std::uint32_t BANK_BYTES        = 16 * 1024;
  static constexpr std::uint32_t STACK_BYTES       = 1024; // reserve 1 KiB for compiler stack
  static constexpr std::uint32_t STACK_BYTES_ALIGN = 1024; // keep B tile starts simple and bank-aligned enough

  static constexpr std::uint32_t A_TILE_BYTES = padded_M * K_SLICE * sizeof(a_t);
  static constexpr std::uint32_t B_TILE_BYTES = K_SLICE * N_SLICE * sizeof(b_t);
  static constexpr std::uint32_t C_TILE_BYTES = padded_M * N_SLICE * sizeof(c_t);

  static_assert(A_TILE_BYTES <= BANK_BYTES,
                "One A ping/pong tile must fit in one 16 KiB bank");
  static_assert(B_TILE_BYTES + STACK_BYTES_ALIGN <= BANK_BYTES,
                "B tile plus reserved stack slack must fit in one 16 KiB bank");
  static_assert(C_TILE_BYTES <= BANK_BYTES,
                "One C ping/pong tile must fit in one 16 KiB bank");

  input_port inA[LHS_PORTS];
  input_port inB[CAS_NUM * CAS_LENGTH];
  output_port outC[CAS_NUM];
  kernel kk[CAS_NUM * CAS_LENGTH];

  static constexpr std::uint32_t bank_base(int bank) {
    return static_cast<std::uint32_t>(bank) * BANK_BYTES;
  }

  void place_graph(int COL_START, int ROW_START)
  {
    for (int idx = 0; idx < CAS_NUM * CAS_LENGTH; ++idx)
    {
      const int tileRow = ROW_START + (idx / CAS_LENGTH);
      const int tileCol = COL_START + (idx % CAS_LENGTH);
      const bool is_last = (idx % CAS_LENGTH) == (CAS_LENGTH - 1);

      adf::location<adf::kernel>(kk[idx]) = adf::tile(tileCol, tileRow);

      const int memCol = tileCol - 1;
      const int memRow = tileRow;

      adf::location<adf::buffer>(kk[idx].in[0]) = {
        adf::bank(memCol, memRow, 0),
        adf::bank(memCol, memRow, 3)
      };

      adf::location<adf::buffer>(kk[idx].in[1]) = {
        adf::bank(tileCol, memRow, 1),
        adf::bank(tileCol, memRow, 2)
      };

      // Stack shares one B bank, but let compiler choose the offset.
      adf::location<adf::stack>(kk[idx]) = adf::bank(tileCol, memRow, 1);

      if (is_last) {
        adf::location<adf::buffer>(kk[idx].out[0]) = {
          adf::bank(tileCol, tileRow, 0),
          adf::bank(tileCol, tileRow, 3)
        };
      }
    }
  }

  matmul_graph() {
    for (int row = 0; row < CAS_NUM; ++row) {
      if constexpr (CAS_LENGTH == 1) {
        kk[row * CAS_LENGTH] = kernel::create_object<matmul_single<ConfigT>>();
      } else {
        kk[row * CAS_LENGTH] = kernel::create_object<matmul_first<ConfigT>>();
        if constexpr (CAS_LENGTH > 2) {
          for (int c = 1; c < CAS_LENGTH - 1; ++c)
            kk[row * CAS_LENGTH + c] = kernel::create_object<matmul_middle<ConfigT>>();
        }
        kk[row * CAS_LENGTH + (CAS_LENGTH - 1)] = kernel::create_object<matmul_last<ConfigT>>();
      }
    }

    for (int idx = 0; idx < CAS_NUM * CAS_LENGTH; ++idx) {
      source(kk[idx]) = "matmul.cpp";
      runtime<ratio>(kk[idx]) = 1.0;
    }

    for (unsigned col = 0; col < CAS_LENGTH; ++col) {
      for (unsigned row = 0; row < CAS_NUM; ++row) {
        const int idx = row * CAS_LENGTH + col;
        connect<>(inA[PARALLELISM_CONTRACT_OUTER ? idx : col], kk[idx].in[0]);
        dimensions(kk[idx].in[0]) = {padded_M * K_SLICE};
      }
    }

    for (unsigned row = 0; row < CAS_NUM; ++row) {
      for (unsigned col = 0; col < CAS_LENGTH; ++col) {
        const int idx = row * CAS_LENGTH + col;
        connect<>(inB[idx], kk[idx].in[1]);
        dimensions(kk[idx].in[1]) = {K_SLICE * N_SLICE};
      }
    }

    for (unsigned row = 0; row < CAS_NUM; ++row) {
      const int last_idx = row * CAS_LENGTH + (CAS_LENGTH - 1);
      connect<>(kk[last_idx].out[0], outC[row]);
      dimensions(kk[last_idx].out[0]) = {padded_M * N_SLICE};
    }

    if constexpr (CAS_LENGTH > 1) {
      for (unsigned row = 0; row < CAS_NUM; ++row) {
        for (unsigned col = 0; col < CAS_LENGTH - 1; ++col) {
          connect<cascade>(kk[row * CAS_LENGTH + col].out[0],
                           kk[row * CAS_LENGTH + col + 1].in[2]);
        }
      }
    }
  }
};
