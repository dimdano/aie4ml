#include "matmul.h"
using namespace adf;

template<typename ConfigT>
matmul_base<ConfigT>::matmul_base() {
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);

  static_assert(
      ConfigT::K_SLICE * ConfigT::N_SLICE * sizeof(typename ConfigT::b_t) <= 16384,
      "B tile size per kernel must fit in one AIE-ML bank (16 KiB)");
  static_assert(
      ConfigT::K_SLICE % (2 * ConfigT::K) == 0,
      "K_SLICE must be divisible by 2*K");
  static_assert(
      ConfigT::N_SLICE % (2 * ConfigT::N) == 0,
      "N_SLICE must be divisible by 2*N");
  static_assert(
      ConfigT::padded_M % (2 * ConfigT::M) == 0,
      "padded_M must be divisible by 2*M");
  static_assert(
      ConfigT::padded_K == ConfigT::K_SLICE * ConfigT::CAS_LENGTH,
      "padded_K must equal K_SLICE * CAS_LENGTH");
  static_assert(
      ConfigT::PARALLELISM_CONTRACT_OUTER
          ? (ConfigT::padded_N == ConfigT::N_SLICE)
          : (ConfigT::padded_N == ConfigT::N_SLICE * ConfigT::CAS_NUM),
      "padded_N must equal N_SLICE ('outer' keeps N whole) or N_SLICE * CAS_NUM ('inner' slices it)");
}

template<typename ConfigT>
void matmul_single<ConfigT>::run(input_buffer<a_t>& A,
                                 input_buffer<b_t>& B,
                                 output_buffer<c_t>& C) {
  static constexpr int rowA  = ConfigT::padded_M;
  static constexpr int colA  = ConfigT::K_SLICE;
  static constexpr int colB  = ConfigT::N_SLICE;
  static constexpr int M     = ConfigT::M;
  static constexpr int K     = ConfigT::K;
  static constexpr int N     = ConfigT::N;
  static constexpr int SHIFT = ConfigT::SHIFT;

  using MMUL = aie::mmul<M, K, N, a_t, b_t, acc_scalar_t>;

  const a_t* pA = A.data();
  const b_t* pB = B.data();
  c_t*       pC = C.data();

  for (unsigned z = 0; z < rowA / M; z += 2) {
    c_t* __restrict pC1 = pC + (      z * (colB / N) + 0) * MMUL::size_C;
    c_t* __restrict pC2 = pC + ((z + 1) * (colB / N) + 0) * MMUL::size_C;

    for (unsigned j = 0; j < colB / N; j += 2) {
      const a_t* __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const a_t* __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;
      const b_t* __restrict pB1 = pB + (0 * (colB / N) +       j) * MMUL::size_B;
      const b_t* __restrict pB2 = pB + (0 * (colB / N) + (j + 1)) * MMUL::size_B;

      aie::vector<a_t, MMUL::size_A> A0, A1;
      if constexpr (ConfigT::TRANSPOSE_A) {
        A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
        A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
      } else {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        A1 = aie::load_v<MMUL::size_A>(pA2);
      }
      pA1 += MMUL::size_A; pA2 += MMUL::size_A;

      aie::vector<b_t, MMUL::size_B> B0, B1;
      if constexpr (ConfigT::TRANSPOSE_B) {
        B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), N, K);
        B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), N, K);
      } else {
        B0 = aie::load_v<MMUL::size_B>(pB1);
        B1 = aie::load_v<MMUL::size_B>(pB2);
      }
      pB1 += MMUL::size_B * (colB / N);
      pB2 += MMUL::size_B * (colB / N);

      MMUL C00; C00.mul(A0, B0);
      MMUL C01; C01.mul(A0, B1);
      MMUL C10; C10.mul(A1, B0);
      MMUL C11; C11.mul(A1, B1);

      for (unsigned i = 1; i < colA / K; ++i)
        chess_prepare_for_pipelining
      {
        if constexpr (ConfigT::TRANSPOSE_A) {
          A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
          A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
        } else {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          A1 = aie::load_v<MMUL::size_A>(pA2);
        }
        pA1 += MMUL::size_A; pA2 += MMUL::size_A;
        if constexpr (ConfigT::TRANSPOSE_B) {
          B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), N, K);
          B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), N, K);
        } else {
          B0 = aie::load_v<MMUL::size_B>(pB1);
          B1 = aie::load_v<MMUL::size_B>(pB2);
        }
        pB1 += MMUL::size_B * (colB / N);
        pB2 += MMUL::size_B * (colB / N);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      aie::store_v(pC1, C00.template to_vector<c_t>(SHIFT)); pC1 += MMUL::size_C;
      aie::store_v(pC1, C01.template to_vector<c_t>(SHIFT)); pC1 += MMUL::size_C;
      aie::store_v(pC2, C10.template to_vector<c_t>(SHIFT)); pC2 += MMUL::size_C;
      aie::store_v(pC2, C11.template to_vector<c_t>(SHIFT)); pC2 += MMUL::size_C;
    }
  }
}

template<typename ConfigT>
void matmul_first<ConfigT>::run(input_buffer<a_t>& A,
                                input_buffer<b_t>& B,
                                output_cascade<acc_scalar_t>* outCascade) {
  static constexpr int rowA = ConfigT::padded_M;
  static constexpr int colA = ConfigT::K_SLICE;
  static constexpr int colB = ConfigT::N_SLICE;
  static constexpr int M    = ConfigT::M;
  static constexpr int K    = ConfigT::K;
  static constexpr int N    = ConfigT::N;

  using MMUL = aie::mmul<M, K, N, a_t, b_t, acc_scalar_t>;

  const a_t* pA = A.data();
  const b_t* pB = B.data();

  for (unsigned z = 0; z < rowA / M; z += 2) {
    for (unsigned j = 0; j < colB / N; j += 2) {
      const a_t* __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const a_t* __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;
      const b_t* __restrict pB1 = pB + (0 * (colB / N) +       j) * MMUL::size_B;
      const b_t* __restrict pB2 = pB + (0 * (colB / N) + (j + 1)) * MMUL::size_B;

      aie::vector<a_t, MMUL::size_A> A0, A1;
      if constexpr (ConfigT::TRANSPOSE_A) {
        A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
        A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
      } else {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        A1 = aie::load_v<MMUL::size_A>(pA2);
      }
      pA1 += MMUL::size_A; pA2 += MMUL::size_A;

      aie::vector<b_t, MMUL::size_B> B0, B1;
      if constexpr (ConfigT::TRANSPOSE_B) {
        B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), N, K);
        B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), N, K);
      } else {
        B0 = aie::load_v<MMUL::size_B>(pB1);
        B1 = aie::load_v<MMUL::size_B>(pB2);
      }
      pB1 += MMUL::size_B * (colB / N);
      pB2 += MMUL::size_B * (colB / N);

      MMUL C00; C00.mul(A0, B0);
      MMUL C01; C01.mul(A0, B1);
      MMUL C10; C10.mul(A1, B0);
      MMUL C11; C11.mul(A1, B1);

      for (unsigned i = 1; i < colA / K; ++i)
        chess_prepare_for_pipelining
      {
        if constexpr (ConfigT::TRANSPOSE_A) {
          A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
          A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
        } else {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          A1 = aie::load_v<MMUL::size_A>(pA2);
        }
        pA1 += MMUL::size_A; pA2 += MMUL::size_A;
        if constexpr (ConfigT::TRANSPOSE_B) {
          B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), N, K);
          B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), N, K);
        } else {
          B0 = aie::load_v<MMUL::size_B>(pB1);
          B1 = aie::load_v<MMUL::size_B>(pB2);
        }
        pB1 += MMUL::size_B * (colB / N);
        pB2 += MMUL::size_B * (colB / N);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      writeincr(outCascade, C00.to_accum());
      writeincr(outCascade, C01.to_accum());
      writeincr(outCascade, C10.to_accum());
      writeincr(outCascade, C11.to_accum());
    }
  }
}

template<typename ConfigT>
void matmul_middle<ConfigT>::run(input_buffer<a_t>& A,
                                 input_buffer<b_t>& B,
                                 input_cascade<acc_scalar_t>* inCascade,
                                 output_cascade<acc_scalar_t>* outCascade) {
  static constexpr int rowA = ConfigT::padded_M;
  static constexpr int colA = ConfigT::K_SLICE;
  static constexpr int colB = ConfigT::N_SLICE;
  static constexpr int M    = ConfigT::M;
  static constexpr int K    = ConfigT::K;
  static constexpr int N    = ConfigT::N;

  using MMUL = aie::mmul<M, K, N, a_t, b_t, acc_scalar_t>;

  const a_t* pA = A.data();
  const b_t* pB = B.data();

  for (unsigned z = 0; z < rowA / M; z += 2) {
    for (unsigned j = 0; j < colB / N; j += 2) {
      auto acc00 = readincr_v<MMUL::size_C>(inCascade);
      auto acc01 = readincr_v<MMUL::size_C>(inCascade);
      auto acc10 = readincr_v<MMUL::size_C>(inCascade);
      auto acc11 = readincr_v<MMUL::size_C>(inCascade);

      MMUL C00; C00 = acc00;
      MMUL C01; C01 = acc01;
      MMUL C10; C10 = acc10;
      MMUL C11; C11 = acc11;

      const a_t* __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const a_t* __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;
      const b_t* __restrict pB1 = pB + (0 * (colB / N) +       j) * MMUL::size_B;
      const b_t* __restrict pB2 = pB + (0 * (colB / N) + (j + 1)) * MMUL::size_B;

      aie::vector<a_t, MMUL::size_A> A0, A1;
      if constexpr (ConfigT::TRANSPOSE_A) {
        A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
        A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
      } else {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        A1 = aie::load_v<MMUL::size_A>(pA2);
      }
      pA1 += MMUL::size_A; pA2 += MMUL::size_A;

      aie::vector<b_t, MMUL::size_B> B0, B1;
      if constexpr (ConfigT::TRANSPOSE_B) {
        B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), N, K);
        B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), N, K);
      } else {
        B0 = aie::load_v<MMUL::size_B>(pB1);
        B1 = aie::load_v<MMUL::size_B>(pB2);
      }
      pB1 += MMUL::size_B * (colB / N);
      pB2 += MMUL::size_B * (colB / N);

      C00.mac(A0, B0);
      C01.mac(A0, B1);
      C10.mac(A1, B0);
      C11.mac(A1, B1);

      for (unsigned i = 1; i < colA / K; ++i)
        chess_prepare_for_pipelining
      {
        if constexpr (ConfigT::TRANSPOSE_A) {
          A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
          A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
        } else {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          A1 = aie::load_v<MMUL::size_A>(pA2);
        }
        pA1 += MMUL::size_A; pA2 += MMUL::size_A;
        if constexpr (ConfigT::TRANSPOSE_B) {
          B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), N, K);
          B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), N, K);
        } else {
          B0 = aie::load_v<MMUL::size_B>(pB1);
          B1 = aie::load_v<MMUL::size_B>(pB2);
        }
        pB1 += MMUL::size_B * (colB / N);
        pB2 += MMUL::size_B * (colB / N);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      writeincr(outCascade, C00.to_accum());
      writeincr(outCascade, C01.to_accum());
      writeincr(outCascade, C10.to_accum());
      writeincr(outCascade, C11.to_accum());
    }
  }
}

template<typename ConfigT>
void matmul_last<ConfigT>::run(input_buffer<a_t>& A,
                               input_buffer<b_t>& B,
                               input_cascade<acc_scalar_t>* inCascade,
                               output_buffer<c_t>& C) {
  static constexpr int rowA  = ConfigT::padded_M;
  static constexpr int colA  = ConfigT::K_SLICE;
  static constexpr int colB  = ConfigT::N_SLICE;
  static constexpr int M     = ConfigT::M;
  static constexpr int K     = ConfigT::K;
  static constexpr int N     = ConfigT::N;
  static constexpr int SHIFT = ConfigT::SHIFT;

  using MMUL = aie::mmul<M, K, N, a_t, b_t, acc_scalar_t>;

  const a_t* pA = A.data();
  const b_t* pB = B.data();
  c_t*       pC = C.data();

  for (unsigned z = 0; z < rowA / M; z += 2) {
    c_t* __restrict pC1 = pC + (      z * (colB / N) + 0) * MMUL::size_C;
    c_t* __restrict pC2 = pC + ((z + 1) * (colB / N) + 0) * MMUL::size_C;

    for (unsigned j = 0; j < colB / N; j += 2) {
      const a_t* __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const a_t* __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;
      const b_t* __restrict pB1 = pB + (0 * (colB / N) +       j) * MMUL::size_B;
      const b_t* __restrict pB2 = pB + (0 * (colB / N) + (j + 1)) * MMUL::size_B;

      MMUL C00(readincr_v<MMUL::size_C>(inCascade));
      MMUL C01(readincr_v<MMUL::size_C>(inCascade));
      MMUL C10(readincr_v<MMUL::size_C>(inCascade));
      MMUL C11(readincr_v<MMUL::size_C>(inCascade));

      aie::vector<a_t, MMUL::size_A> A0, A1;
      if constexpr (ConfigT::TRANSPOSE_A) {
        A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
        A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
      } else {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        A1 = aie::load_v<MMUL::size_A>(pA2);
      }
      pA1 += MMUL::size_A; pA2 += MMUL::size_A;

      aie::vector<b_t, MMUL::size_B> B0, B1;
      if constexpr (ConfigT::TRANSPOSE_B) {
        B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), N, K);
        B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), N, K);
      } else {
        B0 = aie::load_v<MMUL::size_B>(pB1);
        B1 = aie::load_v<MMUL::size_B>(pB2);
      }
      pB1 += MMUL::size_B * (colB / N);
      pB2 += MMUL::size_B * (colB / N);

      C00.mac(A0, B0);  C01.mac(A0, B1);
      C10.mac(A1, B0);  C11.mac(A1, B1);

      for (unsigned i = 1; i < colA / K; ++i)
        chess_prepare_for_pipelining
      {
        if constexpr (ConfigT::TRANSPOSE_A) {
          A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
          A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
        } else {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          A1 = aie::load_v<MMUL::size_A>(pA2);
        }
        pA1 += MMUL::size_A; pA2 += MMUL::size_A;
        if constexpr (ConfigT::TRANSPOSE_B) {
          B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), N, K);
          B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), N, K);
        } else {
          B0 = aie::load_v<MMUL::size_B>(pB1);
          B1 = aie::load_v<MMUL::size_B>(pB2);
        }
        pB1 += MMUL::size_B * (colB / N);
        pB2 += MMUL::size_B * (colB / N);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      aie::store_v(pC1, C00.template to_vector<c_t>(SHIFT)); pC1 += MMUL::size_C;
      aie::store_v(pC1, C01.template to_vector<c_t>(SHIFT)); pC1 += MMUL::size_C;
      aie::store_v(pC2, C10.template to_vector<c_t>(SHIFT)); pC2 += MMUL::size_C;
      aie::store_v(pC2, C11.template to_vector<c_t>(SHIFT)); pC2 += MMUL::size_C;
    }
  }
}
