// Copyright © 2024 Yury Popov (@djphoenix).

#include "ctc_loss/ctc_loss.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace mlx::core {

#define assert_contiguous(a) \
  if (a.strides()[a.ndim()-1] != 1) throw std::runtime_error(#a " should be contiguous on last dimension")

#ifdef _METAL_

static const std::string lib_name = "mlx_ctc";

template<typename ...As>
static inline void dispatch_kernel(
  const Stream &s,
  const std::string &kname,
  MTL::Size grid_size,
  std::initializer_list<const array> inputs,
  std::initializer_list<array> outputs,
  As ...args
) {
  auto& d = metal::device(s.device);
  d.register_library(lib_name, metal::get_colocated_mtllib_path);

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, lib_name);
  compute_encoder->setComputePipelineState(kernel);

  size_t idx = 0;
  for (auto a : inputs ) compute_encoder.set_input_array (a, idx++);
  for (auto a : outputs) compute_encoder.set_output_array(a, idx++);
  (compute_encoder->setBytes(&args, sizeof(As), idx++), ...);

  size_t num_th = kernel->maxTotalThreadsPerThreadgroup();
  MTL::Size group_size;
  group_size.width  = std::min<size_t>(grid_size.width , num_th);
  num_th = std::max<size_t>(1, num_th / group_size.width);
  group_size.height = std::min<size_t>(grid_size.height, num_th);
  num_th = std::max<size_t>(1, num_th / group_size.height);
  group_size.depth  = std::min<size_t>(grid_size.depth , num_th);

  compute_encoder->dispatchThreads(grid_size, group_size);
}

static inline void dispatch_fill_z(const Stream &s, array &a) {
  dispatch_kernel(
    s, "ctc_loss_fill_z_" + type_to_name(a),
    MTL::Size(a.data_size(), 1, 1),
    {}, { a }
  );
}

void CTCLoss::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outarr) {
  auto& log_probs      = inputs[0];
  auto& targets        = inputs[1];
  auto& input_lengths  = inputs[2];
  auto& target_lengths = inputs[3];
  auto& loss           = outarr[0];
  auto& log_alpha      = outarr[1];

  size_t batch_size     = log_probs.shape()[1];
  size_t max_target_len = targets.shape()[1];

  log_alpha.set_data(allocator::malloc_or_wait(log_alpha.nbytes()));
  loss.set_data(allocator::malloc_or_wait(loss.nbytes()));

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
  
  std::string data_type = type_to_name(log_probs);
  std::string indx_type = type_to_name(targets);

  dispatch_kernel(
    stream(),
    "ctc_loss_alpha_" + data_type + "_" + indx_type,
    MTL::Size(max_target_len + 1, batch_size, 1),
    {
      log_probs,
      targets,
      target_lengths,
      input_lengths,
    },
    { log_alpha },
    blank_,
    tgt_stride_B,
    loga_stride_T, loga_stride_B,
    logp_stride_T, logp_stride_B
  );

  dispatch_kernel(
    stream(),
    "ctc_loss_final_" + data_type + "_" + indx_type,
    MTL::Size(batch_size, 1, 1),
    {
      target_lengths,
      input_lengths,
      log_alpha,
    },
    { loss },
    loga_stride_T, loga_stride_B
  );
}

void CTCLossVJP::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outarr) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  d.register_library(lib_name, metal::get_colocated_mtllib_path);

  auto& log_probs      = inputs[0];
  auto& targets        = inputs[1];
  auto& input_lengths  = inputs[2];
  auto& target_lengths = inputs[3];
  auto& log_alpha      = inputs[4];
  auto& nll            = inputs[5];
  auto& ctg            = inputs[6];
  auto& grad           = outarr[0];

  array log_beta (log_alpha.shape(), log_alpha.dtype(), nullptr, {});

  size_t max_input_length = log_probs.shape()[0];
  size_t batch_size       = log_probs.shape()[1];
  size_t max_target_len   = targets  .shape()[1];
  size_t num_channels     = log_probs.shape()[2];

  grad.set_data(allocator::malloc_or_wait(grad.nbytes()));
  log_beta.set_data(allocator::malloc_or_wait(log_beta.nbytes()));

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
  
  std::string data_type = type_to_name(log_probs);
  std::string indx_type = type_to_name(targets);

  dispatch_fill_z(stream(), grad);

  dispatch_kernel(
    stream(),
    "ctc_loss_vjp_" + data_type + "_" + indx_type,
    MTL::Size(max_target_len + 1, batch_size, 1),
    {
      log_probs,
      targets,
      target_lengths,
      input_lengths,
    },
    { log_beta },
    blank_,
    tgt_stride_B,
    logb_stride_T, logb_stride_B,
    logp_stride_T, logp_stride_B
  );

  dispatch_kernel(
    stream(),
    "ctc_loss_vjp_grad_step_" + data_type + "_" + indx_type,
    MTL::Size(batch_size, max_input_length, 1),
    {
      targets,
      target_lengths,
      log_alpha,
      log_beta,
    },
    { grad },
    blank_,
    tgt_stride_B,
    loga_stride_T, loga_stride_B,
    grad_stride_T, grad_stride_B
  );

  dispatch_kernel(
    stream(),
    "ctc_loss_vjp_final_" + data_type + "_" + indx_type,
    MTL::Size(num_channels, batch_size, max_input_length),
    {
      log_probs,
      input_lengths,
      nll, ctg,
    },
    { grad },
    logp_stride_T, logp_stride_B,
    grad_stride_T, grad_stride_B
  );
}

#else // Metal is not available

void CTCLoss::eval_gpu(const std::vector<array>& inputs, std::vector<array>& out) {
  throw std::runtime_error("CTCLoss has no GPU implementation.");
}

void CTCLossVJP::eval_gpu(const std::vector<array>& inputs, std::vector<array>& out) {
  throw std::runtime_error("CTCLossVJP has no GPU implementation.");
}

#endif

} // namespace mlx::core
