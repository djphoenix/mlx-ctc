// Copyright © 2024 Yury Popov (@djphoenix).

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  The Connectionist Temporal Classification loss.
 * 
 *  Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
 *  probability of possible alignments of input to target, producing a loss value which is differentiable
 *  with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
 *  limits the length of the target sequence such that it must be <= the input length.
 * 
 *  Return: `(N)`, where `N = batch size`
 * 
 **/
array ctc_loss(
  /**
   *  The logarithmized probabilities of the outputs (e.g. obtained with `mlx::core::log_softmax`)
   *  of size `(T, N, C)`, where
   *  `T = input length`, `N = batch size`, and
   *  `C = number of classes` (including blank)
   */
  const array& log_probs,
  /**
   *  Target sequences of size `(N, S)`, where
   *  `N = batch size` and `S = max target length`.
   *  Each element in the target sequence is a class index.
   *  Target index cannot be blank (default=0).
   *  Targets are padded to the length of the longest sequence, and stacked.
   */
  const array& targets,
  /**
   *  Lengths of the inputs of size `(N)`, where `N = batch size` (must each be <= `T`).
   *  Lengths are specified for each sequence to achieve masking under the assumption that
   *  sequences are padded to equal lengths.
   */
  const array& input_lengths,
  /**
   *  Lengths of the targets of size `(N)`, where `N = batch size` (must each be <= `S`).
   *  Lengths are specified for each sequence to achieve masking under the assumption that
   *  sequences are padded to equal lengths.
   */
  const array& target_lengths,

  uint64_t blank = 0,   // Blank label, default `0`.
  StreamOrDevice s = {} // Stream on which to schedule the operation
);

class CTCLoss : public Primitive {
private:
  uint64_t blank_;
public:
  explicit CTCLoss(Stream stream, uint64_t blank = 0) : Primitive(stream), blank_(blank) {};
  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& out) override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& out) override;
  void print(std::ostream& os) override { os << "CTCLoss"; }
  bool is_equivalent(const Primitive& other) const override {
    return static_cast<const CTCLoss&>(other).blank_ == blank_;
  }

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;
};

class CTCLossVJP : public Primitive {
private:
  uint64_t blank_;
public:
  explicit CTCLossVJP(Stream stream, uint64_t blank = 0) : Primitive(stream), blank_(blank) {};
  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& out) override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& out) override;
  void print(std::ostream& os) override { os << "CTCLossVJP"; }
  bool is_equivalent(const Primitive& other) const override {
    return static_cast<const CTCLossVJP&>(other).blank_ == blank_;
  }
};

} // namespace mlx::core
