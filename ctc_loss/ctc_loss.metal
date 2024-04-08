// Copyright © 2024 Yury Popov (@djphoenix).

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"

#define MTL_DEVICEP device
#define MTL_CONSTP  constant
namespace stdlib = metal;

#include "ctc_loss_impl.h"

template <typename T>
[[kernel]] void ctc_loss_fill_z(
  device T* v [[buffer(0)]],
  uint x [[thread_position_in_grid]]
) {
  v[x] = neginf<T>;
}

template <typename T, typename I>
[[kernel]] void ctc_loss_alpha(
  device   const      T* log_probs      [[buffer(0)]],
  device   const      I* targets        [[buffer(1)]],
  device   const      I* target_lengths [[buffer(2)]],
  device   const      I* input_lengths  [[buffer(3)]],
  device              T* log_alpha      [[buffer(4)]],
  constant const      I& blank          [[buffer(5)]],
  constant const size_t& tgt_stride_B   [[buffer(6)]],
  constant const size_t& loga_stride_T  [[buffer(7)]],
  constant const size_t& loga_stride_B  [[buffer(8)]],
  constant const size_t& logp_stride_T  [[buffer(9)]],
  constant const size_t& logp_stride_B  [[buffer(10)]],
  uint2 bc [[thread_position_in_grid]]
) {
  size_t b = bc.y;
  size_t c = bc.x;
  size_t target_length = size_t(target_lengths[b]);
  size_t input_length = size_t(input_lengths[b]);
  for (size_t t = 0; t < input_length; t++) {
    metal::threadgroup_barrier(metal::mem_flags::mem_device);
    if (c <= target_length) {
      _ctc_loss_calc_alpha(
        target_lengths,
        targets,
        log_probs,
        log_alpha,
        tgt_stride_B,
        logp_stride_T, logp_stride_B,
        loga_stride_T, loga_stride_B,
        blank,
        t, b, c
      );
    }
  }
}

template <typename T, typename I>
[[kernel]] void ctc_loss_final(
  device   const      I* target_lengths [[buffer(0)]],
  device   const      I* input_lengths  [[buffer(1)]],
  device   const      T* log_alpha      [[buffer(2)]],
  device              T* loss           [[buffer(3)]],
  constant const size_t& loga_stride_T  [[buffer(4)]],
  constant const size_t& loga_stride_B  [[buffer(5)]],
  uint b [[thread_position_in_grid]]
) {
  _ctc_loss_final(
    target_lengths,
    input_lengths,
    log_alpha,
    loss,
    loga_stride_T, loga_stride_B,
    b
  );
}

template <typename T, typename I>
[[kernel]] void ctc_loss_vjp(
  device   const      T* log_probs      [[buffer(0)]],
  device   const      I* targets        [[buffer(1)]],
  device   const      I* target_lengths [[buffer(2)]],
  device   const      I* input_lengths  [[buffer(3)]],
  device              T* log_beta       [[buffer(4)]],
  constant const      I& blank          [[buffer(5)]],
  constant const size_t& tgt_stride_B   [[buffer(6)]],
  constant const size_t& logb_stride_T  [[buffer(7)]],
  constant const size_t& logb_stride_B  [[buffer(8)]],
  constant const size_t& logp_stride_T  [[buffer(9)]],
  constant const size_t& logp_stride_B  [[buffer(10)]],
  uint2 bc [[thread_position_in_grid]]
) {
  size_t b = bc.y;
  size_t c = bc.x;
  size_t input_length = size_t(input_lengths[b]);
  for (size_t t = input_length; t-- > 0;) {
    metal::threadgroup_barrier(metal::mem_flags::mem_device);
    _ctc_loss_vjp_calc_beta(
      input_lengths,
      target_lengths,
      targets,
      log_probs,
      log_beta,
      tgt_stride_B,
      logp_stride_T, logp_stride_B,
      logb_stride_T, logb_stride_B,
      blank,
      t, b, c
    );
  }
}

template <typename T, typename I>
[[kernel]] void ctc_loss_vjp_grad_step(
  device   const      I* targets        [[buffer(0)]],
  device   const      I* target_lengths [[buffer(1)]],
  device   const      T* log_alpha      [[buffer(2)]],
  device   const      T* log_beta       [[buffer(3)]],
  device              T* grad           [[buffer(4)]],
  constant const      I& blank          [[buffer(5)]],
  constant const size_t& tgt_stride_B   [[buffer(6)]],
  constant const size_t& loga_stride_T  [[buffer(7)]],
  constant const size_t& loga_stride_B  [[buffer(8)]],
  constant const size_t& grad_stride_T  [[buffer(9)]],
  constant const size_t& grad_stride_B  [[buffer(10)]],
  uint2 pos [[thread_position_in_grid]]
) {
  _ctc_loss_vjp_grad_step(
    target_lengths,
    targets,
    log_alpha,
    log_beta,
    grad,
    tgt_stride_B,
    loga_stride_T, loga_stride_B,
    loga_stride_T, loga_stride_B,
    grad_stride_T, grad_stride_B,
    blank,
    pos.y, pos.x
  );
}

template <typename T, typename I>
[[kernel]] void ctc_loss_vjp_final(
  device   const      T* log_probs     [[buffer(0)]],
  device   const      I* input_lengths [[buffer(1)]],
  device   const      T* nll           [[buffer(2)]],
  device   const      T* ctg           [[buffer(3)]],
  device              T* grad          [[buffer(4)]],
  constant const size_t& logp_stride_T [[buffer(5)]],
  constant const size_t& logp_stride_B [[buffer(6)]],
  constant const size_t& grad_stride_T [[buffer(7)]],
  constant const size_t& grad_stride_B [[buffer(8)]],
  uint3 pos [[thread_position_in_grid]]
) {
  _ctc_loss_vjp_final(
    input_lengths,
    log_probs,
    nll, ctg,
    grad,
    logp_stride_T, logp_stride_B,
    grad_stride_T, grad_stride_B,
    pos.z, pos.y, pos.x
  );
}

#define inst_fn(base, tname, type, iname, indx, ...)          \
  template [[kernel, host_name(#base "_" #tname "_" #iname)]] \
  void base<type, indx>(__VA_ARGS__)

#define inst_ctc_loss_fill(tname, type) \
  template [[kernel, host_name("ctc_loss_fill_z_" #tname)]] \
  void ctc_loss_fill_z<type>( \
    device type* V [[buffer(0)]], \
    uint x [[thread_position_in_grid]] \
  )

#define inst_ctc_loss_alpha(tname, type, iname, indx)     \
  inst_fn(ctc_loss_alpha, tname, type, iname, indx,       \
    device   const   type* log_probs      [[buffer(0)]],  \
    device   const   indx* targets        [[buffer(1)]],  \
    device   const   indx* target_lengths [[buffer(2)]],  \
    device   const   indx* input_lengths  [[buffer(3)]],  \
    device           type* log_alpha      [[buffer(4)]],  \
    constant const   indx& blank          [[buffer(5)]],  \
    constant const size_t& tgt_stride_B   [[buffer(6)]],  \
    constant const size_t& loga_stride_T  [[buffer(7)]],  \
    constant const size_t& loga_stride_B  [[buffer(8)]],  \
    constant const size_t& logp_stride_T  [[buffer(9)]],  \
    constant const size_t& logp_stride_B  [[buffer(10)]], \
    uint2 bc [[thread_position_in_grid]]                  \
  )

#define inst_ctc_loss_final(tname, type, iname, indx)    \
  inst_fn(ctc_loss_final, tname, type, iname, indx,      \
    device   const   indx* target_lengths [[buffer(0)]], \
    device   const   indx* input_lengths  [[buffer(1)]], \
    device   const   type* log_alpha      [[buffer(2)]], \
    device           type* loss           [[buffer(3)]], \
    constant const size_t& loga_stride_T  [[buffer(4)]], \
    constant const size_t& loga_stride_B  [[buffer(5)]], \
    uint b [[thread_position_in_grid]]                   \
  )

#define inst_ctc_loss_vjp(tname, type, iname, indx)       \
  inst_fn(ctc_loss_vjp, tname, type, iname, indx,         \
    device   const   type* log_probs      [[buffer(0)]],  \
    device   const   indx* targets        [[buffer(1)]],  \
    device   const   indx* target_lengths [[buffer(2)]],  \
    device   const   indx* input_lengths  [[buffer(3)]],  \
    device           type* log_beta       [[buffer(4)]],  \
    constant const   indx& blank          [[buffer(5)]],  \
    constant const size_t& tgt_stride_B   [[buffer(6)]],  \
    constant const size_t& logb_stride_T  [[buffer(7)]],  \
    constant const size_t& logb_stride_B  [[buffer(8)]],  \
    constant const size_t& logp_stride_T  [[buffer(9)]],  \
    constant const size_t& logp_stride_B  [[buffer(10)]], \
    uint2 bc [[thread_position_in_grid]]                  \
  )

#define inst_ctc_loss_vjp_grad_step(tname, type, iname, indx) \
  inst_fn(ctc_loss_vjp_grad_step, tname, type, iname, indx,   \
    device   const   indx* targets        [[buffer(0)]],      \
    device   const   indx* target_lengths [[buffer(1)]],      \
    device   const   type* log_alpha      [[buffer(2)]],      \
    device   const   type* log_beta       [[buffer(3)]],      \
    device           type* grad           [[buffer(4)]],      \
    constant const   indx& blank          [[buffer(5)]],      \
    constant const size_t& tgt_stride_B   [[buffer(6)]],      \
    constant const size_t& loga_stride_T  [[buffer(7)]],      \
    constant const size_t& loga_stride_B  [[buffer(8)]],      \
    constant const size_t& grad_stride_T  [[buffer(9)]],      \
    constant const size_t& grad_stride_B  [[buffer(10)]],     \
    uint2 pos [[thread_position_in_grid]]                     \
  )

#define inst_ctc_loss_vjp_final(tname, type, iname, indx) \
  inst_fn(ctc_loss_vjp_final, tname, type, iname, indx,   \
    device   const   type* log_probs     [[buffer(0)]],   \
    device   const   indx* input_lengths [[buffer(1)]],   \
    device   const   type* nll           [[buffer(2)]],   \
    device   const   type* ctg           [[buffer(3)]],   \
    device           type* grad          [[buffer(4)]],   \
    constant const size_t& logp_stride_T [[buffer(5)]],   \
    constant const size_t& logp_stride_B [[buffer(6)]],   \
    constant const size_t& grad_stride_T [[buffer(7)]],   \
    constant const size_t& grad_stride_B [[buffer(8)]],   \
    uint3 pos [[thread_position_in_grid]]                 \
  )

#define inst_ctc_loss_i(tname, type, iname, indx)        \
  inst_ctc_loss_alpha(tname, type, iname, indx);         \
  inst_ctc_loss_final(tname, type, iname, indx);         \
  inst_ctc_loss_vjp(tname, type, iname, indx);           \
  inst_ctc_loss_vjp_grad_step(tname, type, iname, indx); \
  inst_ctc_loss_vjp_final(tname, type, iname, indx);

#define inst_ctc_loss_all(tname, type)            \
  inst_ctc_loss_i(tname, type, uint64, uint64_t); \
  inst_ctc_loss_i(tname, type,  int64,  int64_t); \
  inst_ctc_loss_i(tname, type, uint32, uint32_t); \
  inst_ctc_loss_i(tname, type,  int32,  int32_t); \
  inst_ctc_loss_i(tname, type, uint16, uint16_t); \
  inst_ctc_loss_i(tname, type,  int16,  int16_t); \
  inst_ctc_loss_i(tname, type,  uint8,  uint8_t); \
  inst_ctc_loss_i(tname, type,   int8,   int8_t); \
  inst_ctc_loss_fill(tname, type);

inst_ctc_loss_all(float32 , float     );
inst_ctc_loss_all(float16 , half      );
inst_ctc_loss_all(bfloat16, bfloat16_t);
