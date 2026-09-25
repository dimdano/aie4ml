#pragma once
#include <adf.h>

// Pins one buffer port where the op contract lists it (see BufferLocation): ping and pong, one per
// bank, relative to the op's anchor.
template<typename PortT, typename LocationT>
void pin_buffer(PortT& port, const LocationT& at, int COL_START, int ROW_START)
{
  adf::location<adf::buffer>(port) = {
    adf::bank(COL_START + at.col, ROW_START + at.row, at.bank0),
    adf::bank(COL_START + at.col, ROW_START + at.row, at.bank1)
  };
}
