// Copyright © 2024 Yury Popov (@djphoenix).

#include "ctc_loss/ctc_loss.h"

namespace mlx::core {

namespace stdlib = std;
#include "ctc_loss_impl.h"

#define assert_contiguous(a) \
  if (a.strides()[a.ndim()-1] != 1) throw std::runtime_error(#a " should be contiguous on last dimension")

template <typename T, typename I>
static void ctc_loss_impl(
  const array& log_probs,
  const array& targets,
  const array& input_lengths,
  const array& target_lengths,
  I blank,
  array& loss,
  array& log_alpha
) {
  size_t input_time_size   = log_probs.shape()[0];
  size_t batch_size        = log_probs.shape()[1];

  loss.set_data(allocator::malloc_or_wait(loss.nbytes()));
  log_alpha.set_data(allocator::malloc_or_wait(log_alpha.nbytes()));

  assert_contiguous(log_probs);
  assert_contiguous(targets);
  assert_contiguous(input_lengths);
  assert_contiguous(target_lengths);
  assert_contiguous(loss);
  assert_contiguous(log_alpha);

  size_t logp_stride_T = log_probs.strides()[0];
  size_t logp_stride_B = log_probs.strides()[1];
  size_t  tgt_stride_B = targets  .strides()[0];
  size_t loga_stride_T = log_alpha.strides()[0];
  size_t loga_stride_B = log_alpha.strides()[1];

  const T* logp_data = log_probs.data<T>();
  const I* tgt_data  = targets.data<I>();
  const I* inl_data  = input_lengths.data<I>();
  const I* tgl_data  = target_lengths.data<I>();
        T* loss_data = loss.data<T>();
        T* loga_data = log_alpha.data<T>();

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t t = 0; t < inl_data[b]; t++) {
      for (size_t c = 0; c <= tgl_data[b]; c++) {
        _ctc_loss_calc_alpha(
          tgl_data,
          tgt_data,
          logp_data,
          loga_data,
          tgt_stride_B,
          logp_stride_T, logp_stride_B,
          loga_stride_T, loga_stride_B,
          blank,
          t, b, c
        );
      }
    }
    _ctc_loss_final(
      tgl_data,
      inl_data,
      loga_data,
      loss_data,
      loga_stride_T, loga_stride_B,
      b
    );
  }
}

template <typename T, typename I>
static void ctc_loss_vjp_impl(
  const array& log_probs,
  const array& targets,
  const array& input_lengths,
  const array& target_lengths,
  const array& log_alpha,
  const array& nll,
  const array& ctg,
  I blank,
  array& grad,
  array& log_beta
) {
  grad.set_data(allocator::malloc_or_wait(grad.nbytes()));
  log_beta.set_data(allocator::malloc_or_wait(log_beta.nbytes()));

  size_t max_input_length  = log_probs.shape()[0];
  size_t batch_size        = log_probs.shape()[1];
  size_t num_channels      = log_probs.shape()[2];

  assert_contiguous(log_probs);
  assert_contiguous(targets);
  assert_contiguous(input_lengths);
  assert_contiguous(target_lengths);
  assert_contiguous(log_alpha);
  assert_contiguous(nll);
  assert_contiguous(ctg);
  assert_contiguous(grad);
  assert_contiguous(log_beta);

  size_t logp_stride_T = log_probs.strides()[0];
  size_t logp_stride_B = log_probs.strides()[1];
  size_t  tgt_stride_B = targets  .strides()[0];
  size_t loga_stride_T = log_alpha.strides()[0];
  size_t loga_stride_B = log_alpha.strides()[1];
  size_t logb_stride_T = log_beta .strides()[0];
  size_t logb_stride_B = log_beta .strides()[1];
  size_t grad_stride_T = grad.strides()[0];
  size_t grad_stride_B = grad.strides()[1];

  const T* logp_data = log_probs.data<T>();
  const I* tgt_data  = targets.data<I>();
  const I* inl_data  = input_lengths.data<I>();
  const I* tgl_data  = target_lengths.data<I>();
  const T* loga_data = log_alpha.data<T>();
  const T* nll_data  = nll.data<T>();
  const T* gro_data  = ctg.data<T>();
        T* grad_data = grad.data<T>();
        T* logb_data = log_beta.data<T>();

  std::fill_n(grad_data, grad.data_size(), neginf<T>);

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t t = inl_data[b]; t-- > 0;) {
      for (size_t s = 0; s <= tgl_data[b]; s++) {
        _ctc_loss_vjp_calc_beta(
          inl_data,
          tgl_data,
          tgt_data,
          logp_data,
          logb_data,
          tgt_stride_B,
          logp_stride_T, logp_stride_B,
          logb_stride_T, logb_stride_B,
          blank,
          t, b, s
        );
      }
      _ctc_loss_vjp_grad_step(
        tgl_data,
        tgt_data,
        loga_data,
        logb_data,
        grad_data,
        tgt_stride_B,
        loga_stride_T, loga_stride_B,
        logb_stride_T, logb_stride_B,
        grad_stride_T, grad_stride_B,
        blank,
        t, b
      );
      for (size_t c = 0; c < num_channels; c++) {
        _ctc_loss_vjp_final(
          inl_data,
          logp_data,
          nll_data,
          gro_data,
          grad_data,
          logp_stride_T, logp_stride_B,
          grad_stride_T, grad_stride_B,
          t, b, c
        );
      }
    }
    for (int t = inl_data[b]; t < max_input_length; t++) {
      std::fill_n(&grad_data[grad_stride_T * t + grad_stride_B * b], num_channels, 0);
    }
  }
}

template <typename T>
static void ctc_loss_impl_i(
  const array& log_probs,
  const array& targets,
  const array& input_lengths,
  const array& target_lengths,
  uint64_t blank,
  array& loss,
  array& log_alpha
) {
  if (targets.dtype() == uint64 || targets.dtype() == int64) {
    return ctc_loss_impl<T, uint64_t>(log_probs, targets, input_lengths, target_lengths, blank, loss, log_alpha);
  }
  if (targets.dtype() == uint32 || targets.dtype() == int32) {
    return ctc_loss_impl<T, uint32_t>(log_probs, targets, input_lengths, target_lengths, blank, loss, log_alpha);
  }
  if (targets.dtype() == uint16 || targets.dtype() == int16) {
    return ctc_loss_impl<T, uint16_t>(log_probs, targets, input_lengths, target_lengths, blank, loss, log_alpha);
  }
  if (targets.dtype() == uint8 || targets.dtype() == int8) {
    return ctc_loss_impl<T, uint8_t>(log_probs, targets, input_lengths, target_lengths, blank, loss, log_alpha);
  }
  throw std::runtime_error("CTCLoss is only supported for integral targets.");
}

template <typename T>
static void ctc_loss_vjp_impl_i(
  const array& log_probs,
  const array& targets,
  const array& input_lengths,
  const array& target_lengths,
  const array& log_alpha,
  const array& nll,
  const array& ctg,
  uint64_t blank,
  array& grad,
  array& log_beta
) {
  if (targets.dtype() == uint64 || targets.dtype() == int64) {
    return ctc_loss_vjp_impl<T, uint64_t>(log_probs, targets, input_lengths, target_lengths, log_alpha, nll, ctg, blank, grad, log_beta);
  }
  if (targets.dtype() == uint32 || targets.dtype() == int32) {
    return ctc_loss_vjp_impl<T, uint32_t>(log_probs, targets, input_lengths, target_lengths, log_alpha, nll, ctg, blank, grad, log_beta);
  }
  if (targets.dtype() == uint16 || targets.dtype() == int16) {
    return ctc_loss_vjp_impl<T, uint16_t>(log_probs, targets, input_lengths, target_lengths, log_alpha, nll, ctg, blank, grad, log_beta);
  }
  if (targets.dtype() == uint8 || targets.dtype() == int8) {
    return ctc_loss_vjp_impl<T, uint8_t>(log_probs, targets, input_lengths, target_lengths, log_alpha, nll, ctg, blank, grad, log_beta);
  }
  throw std::runtime_error("CTCLossVJP is only supported for integral targets.");
}

void CTCLoss::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outarr) {
  auto& log_probs      = inputs[0];
  auto& targets        = inputs[1];
  auto& input_lengths  = inputs[2];
  auto& target_lengths = inputs[3];
  auto& loss           = outarr[0];
  auto& log_alpha      = outarr[1];

  if (loss.dtype() == float32) {
    return ctc_loss_impl_i<float>(log_probs, targets, input_lengths, target_lengths, blank_, loss, log_alpha);
  }
  if (loss.dtype() == float16) {
    return ctc_loss_impl_i<float16_t>(log_probs, targets, input_lengths, target_lengths, blank_, loss, log_alpha);
  }
  if (loss.dtype() == bfloat16) {
    return ctc_loss_impl_i<bfloat16_t>(log_probs, targets, input_lengths, target_lengths, blank_, loss, log_alpha);
  }
  throw std::runtime_error("CTCLoss is only supported for floating point types.");
}

void CTCLossVJP::eval_cpu(const std::vector<array>& inputs, std::vector<array>& outarr) {
  auto& log_probs      = inputs[0];
  auto& targets        = inputs[1];
  auto& input_lengths  = inputs[2];
  auto& target_lengths = inputs[3];
  auto& log_alpha      = inputs[4];
  auto& nll            = inputs[5];
  auto& ctg            = inputs[6];
  auto& grad           = outarr[0];

  array log_beta (log_alpha.shape(), log_alpha.dtype(), nullptr, {});

  if (grad.dtype() == float32) {
    return ctc_loss_vjp_impl_i<float>(log_probs, targets, input_lengths, target_lengths, log_alpha, nll, ctg, blank_, grad, log_beta);
  }
  if (grad.dtype() == float16) {
    return ctc_loss_vjp_impl_i<float16_t>(log_probs, targets, input_lengths, target_lengths, log_alpha, nll, ctg, blank_, grad, log_beta);
  }
  if (grad.dtype() == bfloat16) {
    return ctc_loss_vjp_impl_i<bfloat16_t>(log_probs, targets, input_lengths, target_lengths, log_alpha, nll, ctg, blank_, grad, log_beta);
  }
  throw std::runtime_error("CTCLossVJP is only supported for floating point types.");
}

} // namespace mlx::core
