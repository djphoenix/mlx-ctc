# MLX CTC

[![License](https://img.shields.io/github/license/djphoenix/mlx-ctc)](https://github.com/djphoenix/mlx-ctc/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/mlx-ctc)](https://pypi.org/project/mlx-ctc/)
![PyPI - Status](https://img.shields.io/pypi/status/mlx-ctc)

C++ and Metal extensions for [MLX](https://github.com/ml-explore/mlx) CTC Loss

## Library status

Library is passing initial tests and benchmarks.

However, it is still under development, and so called "alpha".

## Installation

MLX-CTC available on [PyPI](https://pypi.org/project/mlx-ctc/). To install, run:

```bash
pip install mlx-ctc
```

To install latest version from GitHub, run:

```bash
pip install git+https://github.com/djphoenix/mlx-ctc.git@main
```

## Usage

Python API of MLX CTC Loss is designed to completely mimic [pytorch version](https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html).

Minimal usage example for MLX:

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_ctc import ctc_loss

# Target are to be padded
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
input = nn.log_softmax(mx.random.normal((T, N, C)), 2)

# Initialize random batch of targets (0 = blank, 1:C = classes)
target = mx.random.randint(1, C, shape=(N, S), dtype=mx.uint32)
input_lengths = mx.full((N,), T, dtype=mx.uint32)
target_lengths = mx.random.randint(S_min, S, shape=(N,), dtype=mx.uint32)

# Make function that returns loss and gradient
def ctc_loss_mean(i,t,il,tl):
  return (ctc_loss(i,t,il,tl)/tl).mean()
ctc_loss_grad_fn = mx.value_and_grad(ctc_loss_mean)

# Calculate loss and gradient in single call
loss, grad = ctc_loss_grad_fn(input, target, input_lengths, target_lengths)
mx.eval(loss, grad)

print('Loss:', loss.item())
print('Gradient shape:', grad.shape)
```

## TODO

- Add tests and benchmarks
- Optimize code more

## Credits

Thanks [pytorch source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossCTC.cpp) for reference implementation, that used for initial CTC Loss development.
