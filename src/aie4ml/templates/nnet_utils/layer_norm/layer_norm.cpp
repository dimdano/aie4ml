// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm.h"

using namespace adf;

template <typename ConfigT>
layernorm_i8<ConfigT>::layernorm_i8()
    : layernorm_base<ConfigT>()
{}

template <typename ConfigT>
inline __attribute__((always_inline))
int32_t layernorm_i8<ConfigT>::isqrt_q15(int32_t var)
{
    if (var <= 0) return (1 << NORM_SHIFT);

    if constexpr (ConfigT::USE_AIE_INVSQRT) {
        const float inv_f = aie::invsqrt((float)var);
        int32_t q15 = (int32_t)(inv_f * (float)(1 << NORM_SHIFT) + 0.5f);
        if (q15 > (1 << NORM_SHIFT)) q15 = (1 << NORM_SHIFT);
        if (q15 < 0) q15 = 0;
        return q15;
    } else {
        const int lz = clb(var);
        const int k  = 31 - lz;
        const int m  = k >> 1;

        uint32_t mant_q16;
        if (k >= 16) {
            mant_q16 = ((uint32_t)var) >> (k - 16);
        } else {
            mant_q16 = ((uint32_t)var) << (16 - k);
        }

        int idx = (int)((mant_q16 - (1u << 16)) >> 10);
        if (idx > 63) idx = 63;

        const uint16_t seed_u16 = (k & 1)
            ? invsqrt_seed_odd_lut[idx]
            : invsqrt_seed_even_lut[idx];

        int32_t x = ((int32_t)seed_u16) >> m;

        if constexpr (ConfigT::ISQRT_NR_ITERS >= 1) {
            const int32_t x2_q15 = (int32_t)(((int64_t)x * (int64_t)x) >> NORM_SHIFT);
            const int32_t t    = var * x2_q15;
            const int32_t corr = (3 * (1 << NORM_SHIFT)) - t;
            x = (int32_t)(((int64_t)x * (int64_t)corr) >> 16);
        }

        if constexpr (ConfigT::ISQRT_NR_ITERS >= 2) {
            const int32_t x2_q15 = (int32_t)(((int64_t)x * (int64_t)x) >> NORM_SHIFT);
            const int32_t t    = var * x2_q15;
            const int32_t corr = (3 * (1 << NORM_SHIFT)) - t;
            x = (int32_t)(((int64_t)x * (int64_t)corr) >> 16);
        }

        return x;
    }
}

template <typename ConfigT>
inline __attribute__((always_inline))
void layernorm_i8<ConfigT>::layernorm_row(
    const in_t*    __restrict in_ptr,
          out_t*   __restrict out_ptr,
    const int16_t* __restrict gamma_ptr,
    const int16_t* __restrict beta_ptr)
{
    auto vin_it = aie::cbegin_vector<VEC>(in_ptr);

    const aie::vector<int8, VEC> ones8 = aie::broadcast<int8, VEC>(1);
    aie::accum<acc32, VEC> acc_sum = aie::zeros<acc32, VEC>();
    aie::accum<acc32, VEC> acc_sq  = aie::zeros<acc32, VEC>();

    for (int i = 0; i < VECS; ++i)
        chess_prepare_for_pipelining
        chess_loop_range(VECS, VECS)
    {
        const aie::vector<int8, VEC> vx = *vin_it++;
        acc_sum = aie::mac(acc_sum, vx, ones8);
        acc_sq  = aie::mac_square(acc_sq,  vx);
    }

    const int32_t sum_x  = aie::reduce_add(acc_sum.template to_vector<int32>(0));
    const int32_t sum_sq = aie::reduce_add(acc_sq .template to_vector<int32>(0));

    const int32_t mu = (sum_x + (1 << (LOG2_COLS - 1))) >> LOG2_COLS;

    // Numerically stable: sum((x-mu)^2) = sum_sq - 2*mu*sum_x + N*mu^2, then /N.
    // The naive formula (sum_sq/N - mu^2) has catastrophic cancellation for shifted
    // inputs: both floor-divisions accumulate independently, giving up to 2*|mu|
    // units of variance error.
    //
    // Fits in int32: resolver enforces COLS*1B <= bank_bytes
    // (16 KiB), so |sum_sq|, |2*mu*sum_x|, |COLS*mu*mu| each stay under ~512M.
    // Worst-case partial sum < 1B, well below int32 max (2.1B).
    const int32_t centered_sq = sum_sq - 2 * mu * sum_x + COLS * mu * mu;
    int32_t var = centered_sq >> LOG2_COLS;
    if (var < 0) var = 0;

    const int32_t inv_std   = isqrt_q15(var + (int32_t)ConfigT::EPS_Q0);
    const int16_t inv_std16 = (int16_t)std::min((int32_t)32767, inv_std);
    const aie::vector<int16, VEC> inv_std_vec = aie::broadcast<int16, VEC>(inv_std16);
    const aie::vector<int16, VEC> mu_vec      = aie::broadcast<int16, VEC>((int16_t)mu);

    auto vgamma_it = aie::cbegin_vector<VEC>(gamma_ptr);
    auto vbeta_it  = aie::cbegin_vector<VEC>(beta_ptr);
    vin_it = aie::cbegin_vector<VEC>(in_ptr);
    auto vout_it = aie::begin_vector<VEC>(out_ptr);

    for (int i = 0; i < VECS; ++i)
        chess_prepare_for_pipelining
        chess_loop_range(VECS, VECS)
    {
        const aie::vector<int8,  VEC> vx      = *vin_it++;
        const aie::vector<int16, VEC> gamma_v = *vgamma_it++;
        const aie::vector<int16, VEC> beta_v  = *vbeta_it++;

        const aie::vector<int16, VEC> vd16 =
            aie::sub(aie::from_vector<acc32>(vx), mu_vec)
                .template to_vector<int16>(0);

        const aie::accum<acc32, VEC> acc_fs  = aie::mul(inv_std_vec, gamma_v);
        const aie::vector<int16, VEC> fscale = acc_fs.template to_vector<int16>(GAMMA_SHIFT);

        aie::accum<acc32, VEC> acc_out = aie::mul(vd16, fscale);
        acc_out = aie::add(acc_out, beta_v);

        *vout_it++ = acc_out.template to_vector<out_t>(NORM_SHIFT - OUT_SHIFT);
    }
}

template <typename ConfigT>
void layernorm_i8<ConfigT>::run(input_buffer<in_t>&    in,
                                 const int16_t (&gamma)[COLS],
                                 const int16_t (&beta)[COLS],
                                 output_buffer<out_t>&  out)
{
    const in_t*    __restrict in_ptr    = in.data();
          out_t*   __restrict out_ptr   = out.data();
    const int16_t* __restrict gamma_ptr = gamma;
    const int16_t* __restrict beta_ptr  = beta;

    for (int row = 0; row < ROWS; ++row) {
        layernorm_row(in_ptr  + row * COLS,
                      out_ptr + row * COLS,
                      gamma_ptr,
                      beta_ptr);
    }
}
