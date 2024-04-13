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

## Benchmarks

To run benchmark on your machine, use:

```bash
python tests/benchmark.py
```

It will output table with MB/s rates for CPU and GPU runs, compared against PyTorch CPU rate.

Example output (`MNWA3T/A` `Mac14,6`):

```
---------------------------------------------------------------------
| Shape (TxBxCxS)       | Torch MB/s | MLX CPU MB/s | MLX GPU MB/s  |
---------------------------------------------------------------------
|   64 x 128 x 32 x  16 |     174.04 |  241.17 1.4x |  219.95  1.3x |
|  128 x 128 x 32 x  32 |      96.62 |  129.52 1.3x |  702.22  7.3x |
|  256 x 128 x 32 x  64 |      52.67 |   68.18 1.3x |  622.79 11.8x |
|  512 x 128 x 32 x 128 |      25.62 |   34.33 1.3x |  657.06 25.6x |
| 1024 x 128 x 32 x 256 |      13.46 |   18.55 1.4x |  573.43 42.6x |
---------------------------------------------------------------------
|  128 x  32 x 32 x  32 |      99.31 |  139.19 1.4x |  212.03  2.1x |
|  128 x  64 x 32 x  32 |      94.13 |  128.56 1.4x |  421.05  4.5x |
|  128 x 128 x 32 x  32 |      97.42 |  130.74 1.3x |  733.86  7.5x |
|  128 x 256 x 32 x  32 |      94.30 |  125.29 1.3x | 1056.41 11.2x |
|  128 x 512 x 32 x  32 |      95.34 |  123.40 1.3x | 1390.39 14.6x |
---------------------------------------------------------------------
|  128 x 128 x  8 x  32 |      26.00 |   36.64 1.4x |  178.84  6.9x |
|  128 x 128 x 16 x  32 |      52.30 |   68.91 1.3x |  367.45  7.0x |
|  128 x 128 x 32 x  32 |      97.42 |  130.53 1.3x |  720.66  7.4x |
|  128 x 128 x 48 x  32 |     134.16 |  183.62 1.4x |  998.70  7.4x |
|  128 x 128 x 64 x  32 |     168.65 |  231.30 1.4x | 1366.02  8.1x |
---------------------------------------------------------------------
|  256 x 128 x 32 x  16 |     168.38 |  208.64 1.2x |  923.31  5.5x |
|  256 x 128 x 32 x  24 |     116.98 |  144.75 1.2x |  952.47  8.1x |
|  256 x 128 x 32 x  32 |      93.67 |  115.91 1.2x |  914.53  9.8x |
|  256 x 128 x 32 x  48 |      66.15 |   81.21 1.2x |  675.61 10.2x |
|  256 x 128 x 32 x  64 |      53.71 |   70.10 1.3x |  590.71 11.0x |
---------------------------------------------------------------------
```

Please note that MLX does not support multithreading, so we use `torch.set_num_threads(1)` to make comparison fair.

## TODO

- Optimize code more

## Credits

Thanks [pytorch source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossCTC.cpp) for reference implementation, that used for initial CTC Loss development.
