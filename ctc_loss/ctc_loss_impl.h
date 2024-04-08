#ifndef MTL_DEVICEP
#define MTL_DEVICEP
#endif

#ifndef MTL_CONSTP
#define MTL_CONSTP
#endif

template <typename T>
static MTL_CONSTP const T neginf = -stdlib::numeric_limits<T>::infinity();

template <typename T>
static inline T logaddexp(T x, T y) {
  if (x == neginf<T>) return y;
  if (y == neginf<T>) return x;
  T maxval = stdlib::max(x, y);
  T minval = stdlib::min(x, y);
  if (minval == 0) return neginf<T>;
  return maxval + stdlib::log(T(1) + stdlib::exp(minval - maxval));
};

template <typename T>
static inline T logaddexp(T x, T y, T z) {
  if (x == neginf<T>) return logaddexp(y, z);
  if (y == neginf<T>) return logaddexp(x, z);
  if (z == neginf<T>) return logaddexp(x, y);
  T maxval = stdlib::max(x, stdlib::max(y, z));
  return maxval + stdlib::log(stdlib::exp(x - maxval) + stdlib::exp(y - maxval) + stdlib::exp(z - maxval));
};

template<typename T, typename I>
static inline void _ctc_loss_calc_alpha(
  MTL_DEVICEP const I* target_lengths,
  MTL_DEVICEP const I* targets,
  MTL_DEVICEP const T* log_probs,
  MTL_DEVICEP       T* log_alpha,
  size_t tgt_stride_B,
  size_t logp_stride_T, size_t logp_stride_B,
  size_t loga_stride_T, size_t loga_stride_B,
  I blank,
  size_t t, size_t b, size_t c
) {
  size_t target_length = size_t(target_lengths[b]);

  MTL_DEVICEP const I* tgt_batch_data = &targets  [tgt_stride_B * b];
  MTL_DEVICEP const T* logp_time_data = &log_probs[logp_stride_T * (t  ) + logp_stride_B * b];
  MTL_DEVICEP const T* loga_prev_data = &log_alpha[loga_stride_T * (t-1) + loga_stride_B * b];
  MTL_DEVICEP       T* loga_time_data = &log_alpha[loga_stride_T * (t  ) + loga_stride_B * b];

  I ctp = tgt_batch_data[c % target_length];
  I ptp = tgt_batch_data[c-1];

  T p0 = logp_time_data[blank];
  T p1 = logp_time_data[ctp];
  if (t == 0) {
    if (c == 0) {
      loga_time_data[0] = p0;
      loga_time_data[1] = p1;
    } else {
      loga_time_data[c*2+0] = neginf<T>;
      loga_time_data[c*2+1] = neginf<T>;
    }
  } else {
    T a0 = loga_prev_data[c*2+0];
    T a1 = loga_prev_data[c*2+1];
    if (c == 0) {
      loga_time_data[0] = p0 + a0;
      loga_time_data[1] = p1 + logaddexp(a0, a1);
    } else {
      T an = loga_prev_data[c*2-1];
      loga_time_data[c*2+0] = p0 + logaddexp(a0, an);
      loga_time_data[c*2+1] = p1 + ((ctp != ptp) ? logaddexp(a1, a0, an) : logaddexp(a1, a0));
    }
  }
}

template<typename T, typename I>
static inline void _ctc_loss_final(
  MTL_DEVICEP const I* target_lengths,
  MTL_DEVICEP const I* input_lengths,
  MTL_DEVICEP const T* log_alpha,
  MTL_DEVICEP       T* loss,
  size_t loga_stride_T,
  size_t loga_stride_B,
  size_t b
) {
  size_t target_length = size_t(target_lengths[b]);
  size_t input_length = size_t(input_lengths[b]);
  T a0 = log_alpha[loga_stride_T * (input_length-1) + loga_stride_B * b + (target_length*2-1)];
  T a1 = log_alpha[loga_stride_T * (input_length-1) + loga_stride_B * b + (target_length*2  )];
  loss[b] = -logaddexp(a0, a1);
}

template<typename T, typename I>
static inline void _ctc_loss_vjp_calc_beta(
  MTL_DEVICEP const I* input_lengths,
  MTL_DEVICEP const I* target_lengths,
  MTL_DEVICEP const I* targets,
  MTL_DEVICEP const T* log_probs,
  MTL_DEVICEP       T* log_beta,
  size_t tgt_stride_B,
  size_t logp_stride_T, size_t logp_stride_B,
  size_t logb_stride_T, size_t logb_stride_B,
  I blank,
  size_t t, size_t b, size_t s
) {
  size_t input_length  = size_t(input_lengths[b]);
  size_t target_length = size_t(target_lengths[b]);

  MTL_DEVICEP const I* tgt_batch_data = &targets  [tgt_stride_B * b];
  MTL_DEVICEP const T* logp_time_data = &log_probs[logp_stride_T *  t    + logp_stride_B * b];
  MTL_DEVICEP       T* logb_time_data = &log_beta [logb_stride_T *  t    + logb_stride_B * b];

  I ctp = tgt_batch_data[(s  )%target_length];
  I ntp = tgt_batch_data[(s+1)%target_length];
  T p0  = logp_time_data[blank];
  T p1  = logp_time_data[ctp];

  if (t == input_length-1) {
    if (s == target_length-1) {
      logb_time_data[s*2+0] = neginf<T>;
      logb_time_data[s*2+1] = p1;
    } else if (s == target_length) {
      logb_time_data[s*2+0] = p0;
      logb_time_data[s*2+1] = neginf<T>;
    } else {
      logb_time_data[s*2+0] = neginf<T>;
      logb_time_data[s*2+1] = neginf<T>;
    }
    return;
  }

  MTL_DEVICEP const T* logb_prev_data = &log_beta [logb_stride_T * (t+1) + logb_stride_B * b];

  T lb0 = logb_prev_data[s*2+0];
  T lb1 = logb_prev_data[s*2+1];
  logb_time_data[s*2+0] = p0 + logaddexp(lb0, lb1);

  if (s < target_length) {
    T lb2 = logb_prev_data[s*2+2];
    T lb3 = logb_prev_data[s*2+3];
    logb_time_data[s*2+1] = p1 + ((ctp != ntp) ? logaddexp(lb1, lb2, lb3) : logaddexp(lb1, lb2));
  } else {
    logb_time_data[s*2+1] = p1 + lb1;
  }
}

template<typename T, typename I>
static inline void _ctc_loss_vjp_grad_step(
  MTL_DEVICEP const I* target_lengths,
  MTL_DEVICEP const I* targets,
  MTL_DEVICEP const T* log_alpha,
  MTL_DEVICEP const T* log_beta,
  MTL_DEVICEP       T* grad,
  size_t tgt_stride_B,
  size_t loga_stride_T, size_t loga_stride_B,
  size_t logb_stride_T, size_t logb_stride_B,
  size_t grad_stride_T, size_t grad_stride_B,
  I blank,
  size_t t, size_t b
) {
  MTL_DEVICEP const I* tgt_batch_data = &targets  [tgt_stride_B * b];
  MTL_DEVICEP const T* loga_time_data = &log_alpha[loga_stride_T * t + loga_stride_B * b];
  MTL_DEVICEP const T* logb_time_data = &log_beta [logb_stride_T * t + logb_stride_B * b];
  MTL_DEVICEP       T* grad_time_data = &grad     [grad_stride_T * t + grad_stride_B * b];
  size_t target_length = size_t(target_lengths[b]);

  T lcab0 = grad_time_data[blank];
  for (size_t s = 0; s <= target_length; s++) {
    I ctp = tgt_batch_data[s%target_length];
    MTL_DEVICEP T& lcab1 = grad_time_data[ctp];
    lcab0 = logaddexp<T>(lcab0, loga_time_data[s*2+0] + logb_time_data[s*2+0]);
    lcab1 = logaddexp<T>(lcab1, loga_time_data[s*2+1] + logb_time_data[s*2+1]);
  }
  grad_time_data[blank] = lcab0;
}

template<typename T, typename I>
static inline void _ctc_loss_vjp_final(
  MTL_DEVICEP const I* input_lengths,
  MTL_DEVICEP const T* log_probs,
  MTL_DEVICEP const T* loss,
  MTL_DEVICEP const T* grad_out,
  MTL_DEVICEP       T* grad,
  size_t logp_stride_T,
  size_t logp_stride_B,
  size_t grad_stride_T,
  size_t grad_stride_B,
  size_t t, size_t b, size_t c
) {
  size_t input_length  = size_t(input_lengths[b]);
  MTL_DEVICEP const T* logp_time_data = &log_probs[logp_stride_T * t + logp_stride_B * b];
  MTL_DEVICEP       T* grad_time_data = &grad     [grad_stride_T * t + grad_stride_B * b];

  if (t < input_length) {
    T nll = loss[b];
    T gr  = grad_out[b];
    T lp  = logp_time_data[c];
    T res = grad_time_data[c];
    grad_time_data[c] = (stdlib::exp(lp)-stdlib::exp(res + nll - lp)) * gr;
  } else {
    grad_time_data[c] = 0;
  }
}
