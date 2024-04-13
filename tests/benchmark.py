# Copyright © 2024 Yury Popov (@djphoenix).

# Run MLX CTC Loss and gradient benchmarks

import torch
import mlx.core as mx
import mlx.nn as mn
import mlx_ctc
from timeit import timeit

# MLX does not have multithreading support, so limit torch to single thread
torch.set_num_threads(1)

def gen_input(T: int, B: int, C: int, S: int, S_min: int):
  logits = torch.randn(T, B, C).requires_grad_()
  targets = torch.randint(1, C, (B, S), dtype=torch.int)
  input_lengths = torch.randint(T//2, T, (B,), dtype=torch.int)
  target_lengths = torch.randint(S_min, S, (B,), dtype=torch.int)
  return logits, targets, input_lengths, target_lengths

def run_torch(logits: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor):
  loss = torch.nn.functional.ctc_loss(
    logits.log_softmax(dim = -1), targets,
    input_lengths, target_lengths,
    blank=0, reduction='none',
  )
  grad, = torch.autograd.grad(loss.div(target_lengths).mean(), logits, retain_graph = True)
  return loss, grad

mlx_ctc_loss_grad_fn = mx.value_and_grad(lambda p,t,i,l: (mlx_ctc.ctc_loss(mn.log_softmax(p, -1),t,i,l)/l).mean())

def run_mlx(logits: mx.array, targets: mx.array, input_lengths: mx.array, target_lengths: mx.array, stream: mx.Stream):
  with mx.stream(stream):
    loss, grad = mlx_ctc_loss_grad_fn(logits, targets, input_lengths, target_lengths)
    mx.eval(loss, grad)
    return loss, grad

def print_result(head: str, rate_torch: float, rate_mlx_cpu: float, rate_mlx_gpu: float):
  mb = 1024*1024
  print(
    f'| {head:21} '
    f'| {rate_torch/mb:10.2f} '
    f'| {rate_mlx_cpu/mb:7.2f} {rate_mlx_cpu/rate_torch:3.1f}x '
    f'| {rate_mlx_gpu/mb:7.2f} {rate_mlx_gpu/rate_torch:4.1f}x '
    f'|'
  )

def run_bench(T: int, B: int, C: int, S: int, S_min: int, number: int = 50):
  torch_ins = gen_input(T, B, C, S, S_min)
  mlx_ins = tuple(map(mx.array, torch_ins))

  size = torch_ins[0].nelement() * torch_ins[0].element_size()

  rate_torch   = size*number/timeit(lambda: run_torch(*torch_ins),            number=number)
  rate_mlx_cpu = size*number/timeit(lambda: run_mlx(*mlx_ins, stream=mx.cpu), number=number)
  rate_mlx_gpu = size*number/timeit(lambda: run_mlx(*mlx_ins, stream=mx.gpu), number=number)

  print_result(f'{T:4} x {B:3} x {C:2} x {S:3}', rate_torch, rate_mlx_cpu, rate_mlx_gpu)

head = f'| {'Shape (TxBxCxS)':21} | Torch MB/s | MLX CPU MB/s | MLX GPU MB/s  |'
print('-'*len(head))
print(head)
print('-'*len(head))

run_bench(T=  64, B=128, C=32, S= 16, S_min=  8, number=10)
run_bench(T= 128, B=128, C=32, S= 32, S_min= 16, number=10)
run_bench(T= 256, B=128, C=32, S= 64, S_min= 32, number=10)
run_bench(T= 512, B=128, C=32, S=128, S_min= 64, number=10)
run_bench(T=1024, B=128, C=32, S=256, S_min=128, number=10)

print('-'*len(head))

run_bench(T= 128, B= 32, C=32, S= 32, S_min=16, number=10)
run_bench(T= 128, B= 64, C=32, S= 32, S_min=16, number=10)
run_bench(T= 128, B=128, C=32, S= 32, S_min=16, number=10)
run_bench(T= 128, B=256, C=32, S= 32, S_min=16, number=10)
run_bench(T= 128, B=512, C=32, S= 32, S_min=16, number=10)

print('-'*len(head))

run_bench(T= 128, B=128, C= 8, S= 32, S_min=16, number=10)
run_bench(T= 128, B=128, C=16, S= 32, S_min=16, number=10)
run_bench(T= 128, B=128, C=32, S= 32, S_min=16, number=10)
run_bench(T= 128, B=128, C=48, S= 32, S_min=16, number=10)
run_bench(T= 128, B=128, C=64, S= 32, S_min=16, number=10)

print('-'*len(head))

run_bench(T= 256, B=128, C=32, S= 16, S_min= 8, number=10)
run_bench(T= 256, B=128, C=32, S= 24, S_min=12, number=10)
run_bench(T= 256, B=128, C=32, S= 32, S_min=16, number=10)
run_bench(T= 256, B=128, C=32, S= 48, S_min=24, number=10)
run_bench(T= 256, B=128, C=32, S= 64, S_min=32, number=10)

print('-'*len(head))
