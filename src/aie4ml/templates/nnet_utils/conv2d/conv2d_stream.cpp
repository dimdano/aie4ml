// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "conv2d_stream.h"

#include "conv2d_core.h"  // implementations stay out of the header the ADF graph parses

using namespace adf;

// The wire carries the logical tensor: rows, then columns, then channels -- no border, no padded
// channels, no computed-width tail. The kernel keeps a band of the image around instead of the
// whole of it, filling the band from the wire, computing it with the shared core, and sending the
// finished rows on; everything the core needs around the data is built here.
//
// A channel block is eight bytes, which is half of a beat and not a vector on either architecture
// (there is no 64-bit vector register), so a block moves as its two 32-bit words. The words are
// taken from the beat in registers: staging them through memory would cost a round trip and read
// them back through another type.
template<typename ConfigT>
struct conv2d_wire {
  static constexpr int BEAT = 16;  // bytes in one 128-bit stream access
  static constexpr int GROUP = 8;  // one channel block: a pixel's share of one plane
  static constexpr int SPAN = ConfigT::KH;     // rows the window spans (dilation 1)
  static constexpr int BAND = ConfigT::OUT_H;  // output rows per core call
  // A pixel is whole channel blocks and then whatever is left over; a tail only exists when the
  // channel count does not fill a block, and it is the only part the wire moves byte by byte.
  static constexpr int IN_BLOCKS = ConfigT::CIN / GROUP;
  static constexpr int IN_TAIL = ConfigT::CIN % GROUP;
  static constexpr int OUT_BLOCKS = ConfigT::COUT / GROUP;
  static constexpr int OUT_TAIL = ConfigT::COUT % GROUP;
  static constexpr int OUT_BEATS = ConfigT::BANDS * BAND * ConfigT::OUT_W * ConfigT::COUT / BEAT;
};

// A band is a slice of one long run of bytes, so what a band leaves part-way through a beat
// belongs to the next band: the beat state lives across calls, not inside one.
template<typename ConfigT>
struct conv2d_wire_reader {
  using data_t = typename ConfigT::data_t;
  static constexpr int BEAT = conv2d_wire<ConfigT>::BEAT;

  aie::vector<data_t, BEAT> beat;
  int used = BEAT;       // bytes of `beat` already taken (tail path)
  int32 spare[2];        // the half beat a previous band did not use (block path)
  bool held = false;

  // One channel block from wherever the wire has reached, spanning a beat boundary when it must.
  // Only the source offset varies, which a shift can serve; the destination is block-aligned.
  inline void take_group(input_stream<data_t>* wire, data_t* target) {
    aie::vector<data_t, 2 * BEAT> pair;
    if (used + conv2d_wire<ConfigT>::GROUP > BEAT) {
      const aie::vector<data_t, BEAT> next = readincr_v<BEAT>(wire);
      pair = aie::concat(beat, next);
      beat = next;
      used -= BEAT;
    } else {
      pair = aie::concat(beat, beat);
    }
    const auto words = aie::vector_cast<int32>(aie::shuffle_down(pair, used));
    int32* const out = reinterpret_cast<int32*>(target);
    out[0] = words.get(0);
    out[1] = words.get(1);
    used += conv2d_wire<ConfigT>::GROUP;
  }

  inline data_t take_byte(input_stream<data_t>* wire) {
    if (used == BEAT) {
      beat = readincr_v<BEAT>(wire);
      used = 0;
    }
    return beat.get(used++);
  }
};

template<typename ConfigT>
struct conv2d_wire_writer {
  using result_t = typename ConfigT::result_t;
  static constexpr int BEAT = conv2d_wire<ConfigT>::BEAT;

  int32 spare[2];  // a half beat this band could not fill (block path)
  bool held = false;
  result_t staged[BEAT];  // bytes not yet sent (tail path)
  int filled = 0;
  int beats = 0;  // beats sent, which is how the last one is known

  inline void send(output_stream<result_t>* wire, const aie::vector<int32, BEAT / 4>& words) {
    writeincr(wire, aie::vector_cast<result_t>(words), ++beats == conv2d_wire<ConfigT>::OUT_BEATS);
  }

  // A beat's worth of bytes, one at a time: unlike a read, a write cannot shift its destination
  // into place, so a channel tail leaves the whole pixel byte-wise.
  inline void put_byte(output_stream<result_t>* wire, result_t value) {
    staged[filled++] = value;
    if (filled == BEAT) {
      send(wire, aie::vector_cast<int32>(aie::load_v<BEAT>(staged)));
      filled = 0;
    }
  }
};

// Rows of the image into rows of the frame. What surrounds the image keeps the zeros it was given:
// the band shift moves whole frame rows, border included, so only a row past the bottom of the
// image has to be cleared here.
template<typename ConfigT>
static inline void conv2d_wire_rows(input_stream<typename ConfigT::data_t>* wire,
                                    conv2d_wire_reader<ConfigT>& reader,
                                    typename ConfigT::data_t* frame,
                                    int frame_row,
                                    int image_row,
                                    int rows)
{
  using W = conv2d_wire<ConfigT>;
  using G = conv2d_geometry<ConfigT>;
  using data_t = typename ConfigT::data_t;
  const int present = ConfigT::IN_H - image_row < rows ? ConfigT::IN_H - image_row : rows;
  const int loaded = present > 0 ? present : 0;

  data_t* const base = frame + frame_row * G::RB + ConfigT::IN_ORIGIN_C * 8;
  auto target = [&](int index, int blocks) {
    const int pixel = index / blocks;
    return base + (index % blocks) * G::CHB + (pixel / ConfigT::IN_W) * G::RB + (pixel % ConfigT::IN_W) * 8;
  };

  if constexpr (W::IN_TAIL == 0) {
    // Whole blocks back to back, so a beat is two of them and the run of rows is one flat loop:
    // no per-row state, which is what lets it pipeline.
    constexpr int BLOCKS = W::IN_BLOCKS;
    const int groups = loaded * ConfigT::IN_W * BLOCKS;
    int group = 0;
    if (reader.held && groups > 0) {
      int32* const out = reinterpret_cast<int32*>(target(0, BLOCKS));
      out[0] = reader.spare[0];
      out[1] = reader.spare[1];
      reader.held = false;
      group = 1;
    }
    for (; group + 1 < groups; group += 2)
      chess_prepare_for_pipelining
    {
      const auto words = aie::vector_cast<int32>(readincr_v<W::BEAT>(wire));
      int32* const lo = reinterpret_cast<int32*>(target(group, BLOCKS));
      int32* const hi = reinterpret_cast<int32*>(target(group + 1, BLOCKS));
      lo[0] = words.get(0);
      lo[1] = words.get(1);
      hi[0] = words.get(2);
      hi[1] = words.get(3);
    }
    if (group < groups) {  // an odd group count: the other half of the beat is the next band's
      const auto words = aie::vector_cast<int32>(readincr_v<W::BEAT>(wire));
      int32* const lo = reinterpret_cast<int32*>(target(group, BLOCKS));
      lo[0] = words.get(0);
      lo[1] = words.get(1);
      reader.spare[0] = words.get(2);
      reader.spare[1] = words.get(3);
      reader.held = true;
    }
  } else {
    // Channels that do not fill a block: whole blocks still move as blocks, only the leftover
    // channels go one by one, and the rest of their block keeps its zeros.
    for (int row = 0; row < loaded; ++row) {
      for (int column = 0; column < ConfigT::IN_W; ++column) {
        data_t* const pixel = base + row * G::RB + column * 8;
        for (int block = 0; block < W::IN_BLOCKS; ++block)
          reader.take_group(wire, pixel + block * G::CHB);
        for (int channel = 0; channel < W::IN_TAIL; ++channel)
          pixel[W::IN_BLOCKS * G::CHB + channel] = reader.take_byte(wire);
      }
    }
  }

  const auto zero = aie::zeros<data_t, 32>();
  for (int row = loaded; row < rows; ++row) {  // below the image: all border
    for (int block = 0; block < ConfigT::CB; ++block) {
      data_t* const cleared = frame + block * G::CHB + (frame_row + row) * G::RB;
      for (int i = 0; i < G::RB / 32; ++i) aie::store_v(cleared + i * 32, zero);
    }
  }
}

// The band the core just computed, as the tensor: only the logical columns and channels.
template<typename ConfigT>
static inline void conv2d_wire_band(output_stream<typename ConfigT::result_t>* wire,
                                    conv2d_wire_writer<ConfigT>& writer,
                                    const typename ConfigT::result_t* out)
{
  using W = conv2d_wire<ConfigT>;
  using result_t = typename ConfigT::result_t;
  constexpr int ROW = ConfigT::OUT_COLS * 8;
  constexpr int PLANE = ConfigT::OUT_ROWS * ROW;
  const result_t* const image = out + ConfigT::OUT_ORIGIN_R * ROW + ConfigT::OUT_ORIGIN_C * 8;

  if constexpr (W::OUT_TAIL == 0) {
    constexpr int BLOCKS = W::OUT_BLOCKS;
    constexpr int GROUPS = W::BAND * ConfigT::OUT_W * BLOCKS;
    auto source = [&](int index) {
      const int pixel = index / BLOCKS;
      return reinterpret_cast<const int32*>(image + (index % BLOCKS) * PLANE +
                                            (pixel / ConfigT::OUT_W) * ROW + (pixel % ConfigT::OUT_W) * 8);
    };
    aie::vector<int32, W::BEAT / 4> beat;
    int group = 0;
    if (writer.held) {
      const int32* const hi = source(0);
      beat.set(writer.spare[0], 0);
      beat.set(writer.spare[1], 1);
      beat.set(hi[0], 2);
      beat.set(hi[1], 3);
      writer.send(wire, beat);
      writer.held = false;
      group = 1;
    }
    for (; group + 1 < GROUPS; group += 2)
      chess_prepare_for_pipelining
    {
      const int32* const lo = source(group);
      const int32* const hi = source(group + 1);
      beat.set(lo[0], 0);
      beat.set(lo[1], 1);
      beat.set(hi[0], 2);
      beat.set(hi[1], 3);
      writer.send(wire, beat);
    }
    if (group < GROUPS) {  // an odd group count: the next band fills the rest of the beat
      const int32* const lo = source(group);
      writer.spare[0] = lo[0];
      writer.spare[1] = lo[1];
      writer.held = true;
    }
  } else {
    for (int row = 0; row < W::BAND; ++row) {
      for (int column = 0; column < ConfigT::OUT_W; ++column) {
        const result_t* const pixel = image + row * ROW + column * 8;
        for (int block = 0; block < W::OUT_BLOCKS; ++block)
          for (int channel = 0; channel < W::GROUP; ++channel)
            writer.put_byte(wire, pixel[block * PLANE + channel]);
        for (int channel = 0; channel < W::OUT_TAIL; ++channel)
          writer.put_byte(wire, pixel[W::OUT_BLOCKS * PLANE + channel]);
      }
    }
  }
}

template<typename ConfigT>
conv2d_stream<ConfigT>::conv2d_stream() {
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);
  conv2d_check_contract<ConfigT>();
  static_assert(sizeof(typename ConfigT::data_t) == 1 && sizeof(typename ConfigT::result_t) == 1,
                "the stream wrapper moves the wire in bytes; a wider element needs its own beat math");
  static_assert(ConfigT::CAS_LENGTH == 1 && ConfigT::CAS_NUM == 1,
                "a partitioned Conv2D on streams is not implemented yet");
  static_assert(!ConfigT::FILLS_BORDER, "the stream wrapper owns the border of the band it keeps");
  static_assert(ConfigT::IN_ROWS == conv2d_wire<ConfigT>::BAND + ConfigT::KH - 1,
                "the band holds exactly the rows its output rows read");
  // Only the tensor has to be whole beats -- a band may end part-way through one, and the beat
  // state carries the remainder to the next band.
  static_assert(ConfigT::IN_H * ConfigT::IN_W * ConfigT::CIN % conv2d_wire<ConfigT>::BEAT == 0 &&
                    ConfigT::BANDS * conv2d_wire<ConfigT>::BAND * ConfigT::OUT_W * ConfigT::COUT %
                            conv2d_wire<ConfigT>::BEAT ==
                        0,
                "a tensor is a whole number of stream beats");
}

template<typename ConfigT>
void conv2d_stream<ConfigT>::run(input_stream<data_t>* ifm,
                                 const weight_t (&wts)[ConfigT::WN],
                                 const bias_t (&bias)[ConfigT::BN],
                                 output_stream<result_t>* ofm)
{
  using W = conv2d_wire<ConfigT>;
  using G = conv2d_geometry<ConfigT>;
  // The band, not the image: the rows one core call reads, and the rows it writes.
  alignas(32) static data_t frame[ConfigT::IN_BYTES] = {};
  alignas(32) static result_t out[ConfigT::OUT_BYTES];
  conv2d_wire_reader<ConfigT> reader;
  conv2d_wire_writer<ConfigT> writer;

  // The frame outlives the call and the band shift leaves image rows in its top rows, so the top
  // border is rebuilt for every inference. The sides keep the zeros they were given, and the
  // bottom is cleared as rows load.
  const auto zero = aie::zeros<data_t, 32>();
  for (int block = 0; block < ConfigT::CB; ++block)
    for (int row = 0; row < ConfigT::IN_ORIGIN_R; ++row)
      for (int i = 0; i < G::RB / 32; ++i)
        aie::store_v(frame + block * G::CHB + row * G::RB + i * 32, zero);

  conv2d_wire_rows<ConfigT>(ifm, reader, frame, ConfigT::IN_ORIGIN_R, 0, ConfigT::IN_ROWS - ConfigT::IN_ORIGIN_R);
  int image_row = ConfigT::IN_ROWS - ConfigT::IN_ORIGIN_R;

  for (int band = 0; band < ConfigT::BANDS; ++band) {
    conv2d_tile<ConfigT, false, false>(frame, wts, bias, out, nullptr, nullptr);
    conv2d_wire_band<ConfigT>(ofm, writer, out);
    if (band + 1 == ConfigT::BANDS) break;
    // Keep the rows the next band's window still reads, and fill the rest from the wire.
    for (int block = 0; block < ConfigT::CB; ++block) {
      data_t* const plane = frame + block * G::CHB;
      for (int row = 0; row + W::BAND < ConfigT::IN_ROWS; ++row)
        chess_prepare_for_pipelining
      {
        for (int i = 0; i < G::RB / 32; ++i)
          aie::store_v(plane + row * G::RB + i * 32, aie::load_v<32>(plane + (row + W::BAND) * G::RB + i * 32));
      }
    }
    conv2d_wire_rows<ConfigT>(ifm, reader, frame, ConfigT::IN_ROWS - W::BAND, image_row, W::BAND);
    image_row += W::BAND;
  }
}
