#include <vector>

#include "caffe/layers/reverse_time_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReverseTimeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* src = bottom[0]->cpu_data();
  Dtype* const dest = top[0]->mutable_cpu_data();

  // TODO: Remove these tests
  const Dtype* const src_max = src + bottom[0]->count();
  const Dtype* const dest_max = dest + top[0]->count();

  const int count = top[0]->count();
  const int copy_amount = top[0]->count(1);
  const int sub_iter_max = top[0]->shape(0);

  const Dtype* seq_length = bottom[1]->cpu_data();
  const int sub_axis_count = bottom[0]->shape(1);
  const int sub_copy_amount = copy_amount / sub_axis_count;

  for (int n = 0; n < sub_axis_count; ++n) {
    const Dtype* sub_src = src + n * sub_copy_amount;
    const int sub_seq_length = seq_length[n];
    Dtype* target = dest + copy_amount * sub_seq_length - copy_amount
      + n * sub_copy_amount;

    // invert only until sub_seq_length, and copy rest:
    // 1. invert
    for (int t = 0; t < sub_seq_length; ++t) {
      // TODO: remove this test
      DCHECK_GE(sub_src, src);
      DCHECK_GE(target, dest);
      DCHECK_LE(sub_src + sub_copy_amount, src_max);
      DCHECK_LE(target + sub_copy_amount, dest_max);
      caffe_copy(sub_copy_amount, sub_src, target);
      sub_src += copy_amount;
      target -= copy_amount;
    }

    // 2. copy rest
    if (copy_remaining_) {
      Dtype* target = dest + (sub_seq_length) * copy_amount
        + n * sub_copy_amount;

      for (int t = sub_seq_length; t < sub_iter_max; ++t) {
        // TODO: remove this test
        DCHECK_GE(sub_src, src);
        DCHECK_GE(target, dest);
        DCHECK_LE(sub_src + sub_copy_amount, src_max);
        DCHECK_LE(target + sub_copy_amount, dest_max);
        caffe_copy(sub_copy_amount, sub_src, target);
        
        sub_src += copy_amount;
        target += copy_amount;
      }
    }
  }
}

template <typename Dtype>
void ReverseTimeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  const Dtype* src = top[0]->gpu_diff();
  Dtype* const dest = bottom[0]->mutable_gpu_diff();

  // TODO: Remove these tests
  const Dtype* const src_max = src + top[0]->count();
  const Dtype* const dest_max = dest + bottom[0]->count();

  const int count = top[0]->count();
  const int copy_amount = top[0]->count(1);
  const int sub_iter_max = top[0]->shape(0);

  const Dtype* seq_length = bottom[1]->cpu_data();
  const int sub_axis_count = bottom[0]->shape(1);
  const int sub_copy_amount = copy_amount / sub_axis_count;

  for (int n = 0; n < sub_axis_count; ++n) {
    DCHECK_LT(n, bottom[1]->count());
    DCHECK_GE(n, 0);
    const Dtype* sub_src = src + n * sub_copy_amount;
    const int sub_seq_length = seq_length[n];
    Dtype* target = dest + copy_amount * sub_seq_length - copy_amount
      + n * sub_copy_amount;

    // invert only until sub_seq_length, and copy rest:
    // 1. invert
    for (int t = 0; t < sub_seq_length; ++t) {
      // TODO: remove this test
      DCHECK_GE(sub_src, src);
      DCHECK_GE(target, dest);
      DCHECK_LE(sub_src + sub_copy_amount, src_max);
      DCHECK_LE(target + sub_copy_amount, dest_max);
      caffe_copy(sub_copy_amount, sub_src, target);
      sub_src += copy_amount;
      target -= copy_amount;
    }

    // 2. copy rest
    if (copy_remaining_) {
      Dtype* target = dest + (sub_seq_length) * copy_amount
        + n * sub_copy_amount;

      for (int t = sub_seq_length; t < sub_iter_max; ++t) {
        // TODO: remove this test
        DCHECK_GE(sub_src, src);
        DCHECK_GE(target, dest);
        DCHECK_LE(sub_src + sub_copy_amount, src_max);
        DCHECK_LE(target + sub_copy_amount, dest_max);
        caffe_copy(sub_copy_amount, sub_src, target);
        
        sub_src += copy_amount;
        target += copy_amount;
      }
    }
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(ReverseTimeLayer);


}  // namespace caffe
