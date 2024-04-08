import mlx.core as mx

def ctc_loss(
        log_probs: mx.array,
        targets: mx.array,
        input_lengths: mx.array,
        target_lengths: mx.array,
        *,
        blank: int = 0,
        stream: mx.Stream | mx.Device | None = None
    ) -> mx.array:
    """
    The Connectionist Temporal Classification loss
    
    Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
    probability of possible alignments of input to target, producing a loss value which is differentiable
    with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
    limits the length of the target sequence such that it must be <= the input length.
    
    Args:
        log_probs (array):
            The logarithmized probabilities of the outputs (e.g. obtained with `mlx::core::log_softmax`)
            of size `(T, N, C)`, where
            `T = input length`, `N = batch size`, and
            `C = number of classes` (including blank)
        
        targets (array):
            Target sequences of size `(N, S)`, where
            `N = batch size` and `S = max target length`.
            Each element in the target sequence is a class index.
            Target index cannot be blank (default=0).
            Targets are padded to the length of the longest sequence, and stacked.
        
        input_lengths (array):
            Lengths of the inputs of size `(N)`, where `N = batch size` (must each be <= `T`).
            Lengths are specified for each sequence to achieve masking under the assumption that
            sequences are padded to equal lengths.
        
        input_lengths (array):
            Lengths of the targets of size `(N)`, where `N = batch size` (must each be <= `S`).
            Lengths are specified for each sequence to achieve masking under the assumption that
            sequences are padded to equal lengths.
        
        blank (int):
            blank label. Default `0`.
    
    Returns:
        array: `(N)`, where `N = batch size`
    """
    ...
