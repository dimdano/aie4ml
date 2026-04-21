#include "elementwise_add.h"

using namespace adf;

template<typename ConfigT>
elementwise_add_base<ConfigT>::elementwise_add_base()
{
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);
}

template<typename ConfigT>
void elementwise_add_kernel<ConfigT>::run(
    input_buffer<lhs_t>& lhs,
    input_buffer<rhs_t>& rhs,
    output_buffer<result_t>& out)
{
  const lhs_t* __restrict lhs_ptr = lhs.data();
  const rhs_t* __restrict rhs_ptr = rhs.data();
  result_t* __restrict out_ptr = out.data();

  constexpr int VEC = ConfigT::VEC_SIZE;
  constexpr int iters = ConfigT::TILE_ELEMENTS / VEC;

  for (int i = 0; i < iters; ++i)
    chess_prepare_for_pipelining
  {
    aie::vector<lhs_t, VEC> va = aie::load_v<VEC>(lhs_ptr);
    lhs_ptr += VEC;
    aie::vector<rhs_t, VEC> vb = aie::load_v<VEC>(rhs_ptr);
    rhs_ptr += VEC;
    auto acc = aie::add(aie::from_vector<acc_scalar_t>(va), vb);

    aie::store_v(out_ptr, aie::to_vector<result_t>(acc, ConfigT::SHIFT));
    out_ptr += VEC;
  }
}
