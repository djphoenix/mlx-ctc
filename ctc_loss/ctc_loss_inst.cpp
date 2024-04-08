// Copyright © 2024 Yury Popov (@djphoenix).

#include "ctc_loss/ctc_loss.h"

namespace mlx::core {

array ctc_loss(
  const array& log_probs,
  const array& targets,
  const array& input_lengths,
  const array& target_lengths,
  uint64_t blank,
  StreamOrDevice s
) {
  auto out_dtype         = log_probs.dtype();
  auto input_time_size   = log_probs.shape()[0];
  auto batch_size        = log_probs.shape()[1];
  auto input_target_size = targets.shape()[1];

  // Output: loss, log_alpha
  return array::make_arrays(
    { { batch_size }, { input_time_size, batch_size, input_target_size * 2 + 2 } },
    { out_dtype, out_dtype },
    std::make_shared<CTCLoss>(to_stream(s), blank),
    { log_probs, targets, input_lengths, target_lengths }
  )[0];
}

std::vector<array> CTCLoss::vjp(
  const std::vector<array>& primals,
  const std::vector<array>& cotangents,
  const std::vector<int>  & argnums,
  const std::vector<array>& outputs
) {
  auto &log_probs      = primals[0];
  auto &targets        = primals[1];
  auto &input_lengths  = primals[2];
  auto &target_lengths = primals[3];
  auto &nll            = outputs[0];
  auto &log_alpha      = outputs[1];
  auto &ctg            = cotangents[0];

  return { array(
    log_probs.shape(), log_probs.dtype(),
    std::make_shared<CTCLossVJP>(stream(), blank_),
    { log_probs, targets, input_lengths, target_lengths, log_alpha, nll, ctg }
  ) };
}

} // namespace mlx::core
