# Copyright © 2024 Yury Popov (@djphoenix).

# Check MLX CTC Loss and gradient output against pytorch

import torch
import mlx.core as mx
import mlx.nn as mn
import numpy as np
import mlx_ctc

# 1. Generate input

T, B, C = 128, 256, 32
t = T // 4
test_time = 10

logits = torch.randn(T, B, C).requires_grad_()
targets = torch.randint(1, C, (B, t), dtype=torch.int16)
input_lengths = torch.randint(T//2, T, (B,), dtype=torch.int16)
target_lengths = torch.randint(t//2, t, (B,), dtype=torch.int16)

print('Logits shape (time X batch X channels):', 'x'.join(map(str, logits.shape)))

mx_logits = mx.array(logits)
mx_targets = mx.array(targets)
mx_target_lengths = mx.array(target_lengths)
mx_input_lengths = mx.array(input_lengths)

# 2. Generate reference output

ref_ctc = torch.nn.functional.ctc_loss(
  logits.log_softmax(dim = -1), targets,
  input_lengths, target_lengths,
  blank=0, reduction='none',
)
ref_grad, = torch.autograd.grad(ref_ctc.div(target_lengths).mean(), logits, retain_graph = True)

# 3. Generate and verify MLX output

mx_ctc_loss_grad = mx.value_and_grad(lambda p,t,i,l: (((x := mlx_ctc.ctc_loss(mn.log_softmax(p, -1),t,i,l))/l).mean(), x))

with mx.stream(mx.cpu):
  (_, mlx_ctc_loss), mlx_ctc_grad = mx_ctc_loss_grad(mx_logits, mx_targets, mx_input_lengths, mx_target_lengths)
  mx.eval(mlx_ctc_loss, mlx_ctc_grad)
  print('CPU Loss diff', torch.sub(ref_ctc .detach(), torch.tensor(np.array(mlx_ctc_loss))).abs().div(ref_ctc .abs().max()).max().item())
  print('CPU Grad diff', torch.sub(ref_grad.detach(), torch.tensor(np.array(mlx_ctc_grad))).abs().div(ref_grad.abs().max()).max().item())

with mx.stream(mx.gpu):
  (_, mlx_ctc_loss), mlx_ctc_grad = mx_ctc_loss_grad(mx_logits, mx_targets, mx_input_lengths, mx_target_lengths)
  mx.eval(mlx_ctc_loss, mlx_ctc_grad)
  print('GPU Loss diff', torch.sub(ref_ctc .detach(), torch.tensor(np.array(mlx_ctc_loss))).abs().div(ref_ctc .abs().max()).max().item())
  print('GPU Grad diff', torch.sub(ref_grad.detach(), torch.tensor(np.array(mlx_ctc_grad))).abs().div(ref_grad.abs().max()).max().item())
