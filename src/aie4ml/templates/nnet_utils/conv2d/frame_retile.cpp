// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include <numeric>
#include "frame_retile.h"

using namespace adf;

namespace {
// An 8-byte channel block is not a vector on any generation, so it moves as two words; naming
// the word type `may_alias` makes that defined on an int8 buffer.
using retile_word_t __attribute__((may_alias)) = int32;
}  // namespace

template<typename ConfigT>
frame_retile<ConfigT>::frame_retile() {
  constexpr int H = ConfigT::SRC_H, W = ConfigT::SRC_W, C = ConfigT::SRC_C;
  constexpr bool LINEAR = ConfigT::SRC_BASE == 0 && ConfigT::SRC_PIXEL == C && ConfigT::SRC_ROW == W * C &&
                          ConfigT::SRC_BLOCK == 8;
  constexpr bool BLOCKED = C % 8 == 0 && ConfigT::SRC_PIXEL == 8 && ConfigT::SRC_ROW >= W * 8 &&
                           ConfigT::SRC_BLOCK >= H * ConfigT::SRC_ROW && ConfigT::SRC_BASE % 8 == 0;
  static_assert(LINEAR || BLOCKED, "the source is a linear tensor or a channel-blocked frame");
  static_assert(sizeof(data_t) == 1, "the frame holds 8-channel blocks of bytes");
  static_assert(ConfigT::SRC_BASE + (ConfigT::CB - 1) * ConfigT::SRC_BLOCK + (H - 1) * ConfigT::SRC_ROW +
                    (W - 1) * ConfigT::SRC_PIXEL + (C - (ConfigT::CB - 1) * 8) <= ConfigT::SRC_BYTES,
                "the source holds the whole image");
  static_assert(ConfigT::CB * 8 >= ConfigT::SRC_C && ConfigT::CB * 8 < ConfigT::SRC_C + 8,
                "the frame has exactly the channel blocks the tensor needs");
  static_assert(ConfigT::COLS % ConfigT::STRIDE == 0, "frame columns divide into phases");
  static_assert(ConfigT::ORIGIN_C + ConfigT::SRC_W <= ConfigT::COLS, "the tensor fits the frame columns");
  static_assert(ConfigT::ORIGIN_R < ConfigT::ROWS, "the tensor starts inside the frame");
  static_assert(ConfigT::FRAME_BYTES == ConfigT::CB * ConfigT::ROWS * ConfigT::COLS * 8, "frame size");
  static_assert(ConfigT::FRAME_BYTES % 16 == 0, "the frame clears in whole vectors");
}

template<typename ConfigT>
void frame_retile<ConfigT>::run(input_buffer<data_t>& src, output_buffer<data_t>& frame)
{
  constexpr int S = ConfigT::STRIDE, W = ConfigT::SRC_W, C = ConfigT::SRC_C;
  constexpr int PW = ConfigT::COLS / S;
  constexpr int RB = ConfigT::COLS * 8, CHB = ConfigT::ROWS * RB;
  constexpr int AVAIL = ConfigT::ROWS - ConfigT::ORIGIN_R;
  constexpr int ROWS = AVAIL < ConfigT::SRC_H ? AVAIL : ConfigT::SRC_H;  // image rows the frame holds
  const data_t* __restrict in = src.data() + ConfigT::SRC_BASE;
  data_t* __restrict out = frame.data();

  // The frame is written whole every call: an output buffer alternates and holds nothing the last
  // call left, so the border and the unused channel lanes are cleared here, not assumed.
  const auto zero = aie::zeros<data_t, 16>();
  for (int i = 0; i < ConfigT::FRAME_BYTES / 16; ++i)
    chess_prepare_for_pipelining
  {
    aie::store_v(out + i * 16, zero);
  }

  if constexpr (S == 2 && ConfigT::SRC_PIXEL == 8 && ConfigT::ORIGIN_C % 2 == 0) {
    // Four pixels a load; even ones go to phase 0 and odd ones to phase 1, one 16-byte store each.
    // Those stores are aligned only where the frame column is a multiple of four, so the pixels
    // before the first such column, and those after the last whole chunk, move one at a time. The
    // counts are constants, so their loops unroll and the loop over rows pipelines. A source row
    // starts only as aligned as the steps that reach it, which the loads are told.
    static_assert((PW * 8) % 16 == 0 && RB % 16 == 0, "both phases of every row start on a vector");
    constexpr int BLOCK_STEP = ConfigT::CB > 1 ? ConfigT::SRC_BLOCK : 0;  // no step with one block
    constexpr unsigned SRC_ALIGN =
      std::gcd(std::gcd(ConfigT::SRC_BASE, ConfigT::SRC_ROW), std::gcd(BLOCK_STEP, 16));
    constexpr int ALIGN = (4 - ConfigT::ORIGIN_C % 4) % 4;
    constexpr int LEAD = W < ALIGN ? W : ALIGN;
    constexpr int CHUNKS = (W - LEAD) / 4, TAIL = (W - LEAD) % 4;
    auto move_pixel = [&](data_t* row, const data_t* src, int c) {
      const int col = ConfigT::ORIGIN_C + c;
      const retile_word_t* sw = reinterpret_cast<const retile_word_t*>(src + c * 8);
      retile_word_t* dw = reinterpret_cast<retile_word_t*>(row + (col % 2) * PW * 8 + (col / 2) * 8);
      dw[0] = sw[0];
      dw[1] = sw[1];
    };
    for (int cb = 0; cb < ConfigT::CB; ++cb)
    for (int r = 0; r < ROWS; ++r)
      chess_prepare_for_pipelining
    {
      const data_t* const src = in + cb * ConfigT::SRC_BLOCK + r * ConfigT::SRC_ROW;
      data_t* const row = out + cb * CHB + (ConfigT::ORIGIN_R + r) * RB;
      for (int c = 0; c < LEAD; ++c)
        chess_flatten_loop
      {
        move_pixel(row, src, c);
      }
      for (int k = 0; k < CHUNKS; ++k)
        chess_flatten_loop
      {
        const int c = LEAD + 4 * k;
        const int col = ConfigT::ORIGIN_C + c;  // a multiple of four: both stores are aligned
        const aie::vector<data_t, 32> v = aie::load_unaligned_v<32>(src + c * 8, SRC_ALIGN);
        aie::store_v(row + (col / 2) * 8, aie::filter_even(v, 8));
        aie::store_v(row + PW * 8 + (col / 2) * 8, aie::filter_odd(v, 8));
      }
      for (int t = 0; t < TAIL; ++t)
        chess_flatten_loop
      {
        move_pixel(row, src, LEAD + 4 * CHUNKS + t);
      }
    }
  } else {
    // One phase at a time, so the frame side advances one pixel slot per step.
    for (int r = 0; r < ROWS; ++r) {
      for (int p = 0; p < S; ++p) {
        const int col = ConfigT::ORIGIN_C + p;
        const int n = (W - p + S - 1) / S;
        for (int cb = 0; cb < ConfigT::CB; ++cb) {
          const data_t* s = in + cb * ConfigT::SRC_BLOCK + r * ConfigT::SRC_ROW + p * ConfigT::SRC_PIXEL;
          data_t* d = out + cb * CHB + (ConfigT::ORIGIN_R + r) * RB + (col % S) * PW * 8 + (col / S) * 8;
          if constexpr (C % 8 == 0) {  // whole blocks, and every pixel starts word-aligned
            for (int i = 0; i < n; ++i)
              chess_prepare_for_pipelining
            {
              const retile_word_t* sw = reinterpret_cast<const retile_word_t*>(s + i * S * ConfigT::SRC_PIXEL);
              retile_word_t* dw = reinterpret_cast<retile_word_t*>(d + i * 8);
              dw[0] = sw[0];
              dw[1] = sw[1];
            }
          } else {
            // The last block of a channel count that does not fill it leaves its other lanes zero.
            const int lanes = C - cb * 8 < 8 ? C - cb * 8 : 8;
            for (int i = 0; i < n; ++i)
              chess_prepare_for_pipelining
            {
              for (int k = 0; k < lanes; ++k) d[i * 8 + k] = s[i * S * ConfigT::SRC_PIXEL + k];
            }
          }
        }
      }
    }
  }
}
