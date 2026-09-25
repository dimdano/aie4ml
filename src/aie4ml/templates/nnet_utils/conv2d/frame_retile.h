// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

// Builds a strided Conv2D's input frame, and nothing else: the zero-bordered, channel-blocked,
// column-polyphase frame [CB][ROWS][STRIDE][COLS / STRIDE][8] the conv compute core reads when a
// strided window's taps must stay contiguous. Two sources are supported: the linear NHWC tensor a
// graph boundary carries, or the channel-blocked frame [CB][rows][cols][8] a producing kernel
// writes, image at some origin.
// ConfigT declares:
//   data_t
//   SRC_H, SRC_W, SRC_C   the image, and the channels each pixel moves
//   SRC_BYTES             bytes per inference in the source buffer; anything past the image is
//                         padding or border and is never read
//   SRC_BASE              where the image's first pixel starts
//   SRC_PIXEL, SRC_ROW, SRC_BLOCK   byte steps between adjacent columns, rows and 8-channel blocks
//   ROWS, COLS, CB        the frame; it may hold fewer rows than the image, never more
//   STRIDE                column phases
//   ORIGIN_R, ORIGIN_C    where the image starts inside the frame's border
//   FRAME_BYTES           = CB * ROWS * COLS * 8

#pragma once
#include <adf.h>
#include <aie_api/aie.hpp>
#include "parameters.h"

using namespace adf;

template<typename ConfigT>
class frame_retile {
public:
  using data_t = typename ConfigT::data_t;

  frame_retile();
  void run(input_buffer<data_t>& src, output_buffer<data_t>& frame);

  static void registerKernelClass() { REGISTER_FUNCTION(frame_retile::run); }
};
