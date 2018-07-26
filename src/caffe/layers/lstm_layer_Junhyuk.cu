#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/lstm_layer_Junhyuk.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype sigmoid(const Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename Dtype>
__device__ Dtype tanh(const Dtype x) {
  return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
}

template <typename Dtype>
__global__ void ClipAdd(const int nthreads, const int dim, int t,
    const Dtype* clip, const Dtype* add_vec, Dtype* data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const Dtype clip_t = clip ? clip[n] : Dtype(t > 0);
    data[index] += clip_t * add_vec[index];
  }
}

template <typename Dtype>
__global__ void ActivationForward(const int nthreads, const int H,
                                const Dtype* pre_gate, Dtype* gate) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % (4*H);
    gate[index] = d < 3*H ? sigmoid(pre_gate[index]) : tanh(pre_gate[index]);
  }
}

template <typename Dtype>
__global__ void LSTMForward(const int nthreads, const int H, const int t,
    const Dtype* c_prev, const Dtype* gate, const Dtype* clip,
    Dtype* c_t, Dtype* h_t) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / H;
    const int d = index % H;
    const Dtype* offset = gate + 4*H*n;
    const Dtype i_t = offset[d];
    const Dtype f_t = offset[H + d];
    const Dtype o_t = offset[2*H + d];
    const Dtype g_t = offset[3*H + d];
    const Dtype c_t_1 = c_prev[index];
    const Dtype clip_t = clip ? clip[n] : Dtype(t > 0);
    c_t[index] = clip_t * f_t * c_t_1 + i_t * g_t;
    h_t[index] = o_t * tanh(c_t[index]);
  }
}

template <typename Dtype>
__global__ void LSTMBackward(const int nthreads, const int H, const int t, 
    const Dtype* c_prev, const Dtype* gate, const Dtype* c_t, 
    const Dtype* clip, Dtype* dc_t, const Dtype* dh_t, 
    Dtype* dc_prev, Dtype* gate_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / H;
    const int d = index % H;
    const Dtype* gate_t = gate + 4*H*n;
    const Dtype i_t = gate_t[d];
    const Dtype f_t = gate_t[H + d];
    const Dtype o_t = gate_t[2*H + d];
    const Dtype g_t = gate_t[3*H + d];
    const Dtype c_t_1 = c_prev[index];
    const Dtype c = c_t[index];
    const Dtype tanh_c = tanh(c);
    const Dtype clip_t = clip ? clip[n] : Dtype(t > 0);
    Dtype* dc_t_1 = dc_prev + index;
    Dtype* gate_diff_t = gate_diff + 4*H*n;
    Dtype* di_t = gate_diff_t + d;
    Dtype* df_t = gate_diff_t + H + d;
    Dtype* do_t = gate_diff_t + 2*H + d;
    Dtype* dg_t = gate_diff_t + 3*H + d;
    
    // Output gate : tanh(c(t)) * h_diff(t)
    *do_t = dh_t[index] * tanh_c;
    // Cell state : o(t) * tanh'(c(t)) * h_diff(t) + f(t+1) * c_diff(t+1)
    dc_t[index] += dh_t[index] * o_t * (Dtype(1) - tanh_c * tanh_c);
    // c_diff(t-1) += f(t) * c_diff(t)
    *dc_t_1 = clip_t * dc_t[index] * f_t;
    // Forget gate : c(t-1) * c_diff(t)
    *df_t = clip_t * dc_t[index] * c_t_1;
    // Input gate : g(t) * c_diff(t)
    *di_t = dc_t[index] * g_t;
    // Input modulation gate : i(t) * c_diff(t)
    *dg_t = dc_t[index] * i_t;
  }
}

template <typename Dtype>
__global__ void ActivationBackward(const int nthreads, const int H, 
    const Dtype clip_threshold, const Dtype* gate, const Dtype* gate_diff, 
    Dtype* pre_gate_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % (4 * H);
    const Dtype gate_val = gate[index];
    if (d < 3 * H) {
      pre_gate_diff[index] = gate_diff[index] * gate_val * (Dtype(1) - gate_val);
    } else {
      pre_gate_diff[index] = gate_diff[index] * (Dtype(1) - gate_val * gate_val);
    }
    if (clip_threshold > Dtype(0)) {
      if (pre_gate_diff[index] < -clip_threshold) {
        pre_gate_diff[index] = -clip_threshold;
      }
      else if (pre_gate_diff[index] > clip_threshold) {
        pre_gate_diff[index] = clip_threshold;
      }
    }
  }
}

template <typename Dtype>
void LstmLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top[0]->gpu_data(), top_.gpu_data());
  Dtype* top_data = top_.mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* clip = NULL;
  if (bottom.size() > 1) {
    clip = bottom[1]->gpu_data();
    CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  }
  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();
  const Dtype* bias = this->blobs_[2]->gpu_data();
  Dtype* pre_gate_data = pre_gate_.mutable_gpu_data();
  Dtype* gate_data = gate_.mutable_gpu_data();
  Dtype* cell_data = cell_.mutable_gpu_data();

  // Initialize previous state
  if (clip) {
    caffe_copy(c_0_.count(), c_T_.gpu_data(), c_0_.mutable_gpu_data());
    caffe_copy(h_0_.count(), h_T_.gpu_data(), h_0_.mutable_gpu_data());
  }
  else {
    caffe_gpu_set(c_0_.count(), Dtype(0.), c_0_.mutable_gpu_data());
    caffe_gpu_set(h_0_.count(), Dtype(0.), h_0_.mutable_gpu_data());
  }

  // Compute input to hidden forward propagation
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, T_*N_, 4*H_, I_, Dtype(1.),
      bottom_data, weight_i, Dtype(0.), pre_gate_data);
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, 4*H_, 1, Dtype(1.),
      bias_multiplier_.gpu_data(), bias, Dtype(1.), pre_gate_data);

  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) {
    Dtype* h_t = top_data + top_.offset(t);
    Dtype* c_t = cell_data + cell_.offset(t);
    Dtype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);
    Dtype* gate_t = gate_data + gate_.offset(t);
    const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
    const Dtype* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.gpu_data();
    const Dtype* c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.gpu_data();

    caffe_gpu_gemm(CblasNoTrans, CblasTrans, N_, 4*H_, H_, Dtype(1.), 
        h_t_1, weight_h, Dtype(0.), h_to_gate_.mutable_gpu_data());
    ClipAdd<Dtype><<<CAFFE_GET_BLOCKS(4*N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        4*N_*H_, 4*H_, t, clip_t, h_to_gate_.gpu_data(), pre_gate_t);
    CUDA_POST_KERNEL_CHECK;
    ActivationForward<Dtype><<<CAFFE_GET_BLOCKS(4*N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        4*N_*H_, H_, pre_gate_t, gate_t);
    CUDA_POST_KERNEL_CHECK;
    LSTMForward<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, H_, t, c_t_1, gate_t, clip_t, c_t, h_t);
    CUDA_POST_KERNEL_CHECK;
  }

  // Preserve cell state and output value for truncated BPTT
  caffe_copy(N_*H_, cell_data + cell_.offset(T_-1), c_T_.mutable_gpu_data());
  caffe_copy(N_*H_, top_data + top_.offset(T_-1), h_T_.mutable_gpu_data());
}

template <typename Dtype>
void LstmLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top_.gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* clip = NULL;
  if (bottom.size() > 1) {
    clip = bottom[1]->gpu_data();
    CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  }
  const Dtype* weight_i = this->blobs_[0]->gpu_data();
  const Dtype* weight_h = this->blobs_[1]->gpu_data();
  const Dtype* gate_data = gate_.gpu_data();
  const Dtype* cell_data = cell_.gpu_data();

  Dtype* top_diff = top_.mutable_gpu_diff();
  Dtype* pre_gate_diff = pre_gate_.mutable_gpu_diff();
  Dtype* gate_diff = gate_.mutable_gpu_diff();
  Dtype* cell_diff = cell_.mutable_gpu_diff();

  caffe_copy(N_*H_, c_T_.gpu_diff(), cell_diff + cell_.offset(T_-1));  

  for (int t = T_-1; t >= 0; --t) {
    Dtype* dh_t = top_diff + top_.offset(t);
    Dtype* dc_t = cell_diff + cell_.offset(t);
    Dtype* gate_diff_t = gate_diff + gate_.offset(t);
    Dtype* pre_gate_diff_t = pre_gate_diff + pre_gate_.offset(t);
    Dtype* dh_t_1 = t > 0 ? top_diff + top_.offset(t-1) : h_0_.mutable_gpu_diff();
    Dtype* dc_t_1 = t > 0 ? cell_diff + cell_.offset(t-1) : c_0_.mutable_gpu_diff();
    const Dtype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
    const Dtype* c_t = cell_data + cell_.offset(t);
    const Dtype* c_t_1 = t > 0 ? cell_data + cell_.offset(t-1) : c_0_.gpu_data();
    const Dtype* gate_t = gate_data + gate_.offset(t);

    LSTMBackward<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, H_, t, c_t_1, gate_t, c_t, clip_t, dc_t, dh_t, dc_t_1, gate_diff_t);
    CUDA_POST_KERNEL_CHECK;
    ActivationBackward<Dtype><<<CAFFE_GET_BLOCKS(4*N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        4*N_*H_, H_, clipping_threshold_, gate_t, gate_diff_t, pre_gate_diff_t);
    CUDA_POST_KERNEL_CHECK;
    
    // Backprop errors to the previous time step
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, N_, H_, 4*H_,
        Dtype(1.), pre_gate_diff_t, weight_h, Dtype(0.), h_to_h_.mutable_gpu_data());
    ClipAdd<Dtype><<<CAFFE_GET_BLOCKS(N_*H_), CAFFE_CUDA_NUM_THREADS>>>(
        N_*H_, H_, t, clip_t, h_to_h_.gpu_data(), dh_t_1);
  }
 
  if (this->param_propagate_down_[0]) {
    // Gradient w.r.t. input-to-hidden weight
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, 4*H_, I_, T_*N_, Dtype(1.),
        pre_gate_diff, bottom_data, Dtype(1.), this->blobs_[0]->mutable_gpu_diff());
  }

  if (this->param_propagate_down_[1]) {
    // Gradient w.r.t. hidden-to-hidden weight
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, 4*H_, H_, (T_-1)*N_, Dtype(1.),
        pre_gate_diff + pre_gate_.offset(1), top_data, 
        Dtype(1.), this->blobs_[1]->mutable_gpu_diff());

    // Add Gradient from previous time-step
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, 4*H_, H_, 1, Dtype(1.),
        pre_gate_diff, h_0_.gpu_data(), 
        Dtype(1.), this->blobs_[1]->mutable_gpu_diff());
  }
  if (this->param_propagate_down_[2]) { 
    // Gradient w.r.t. bias
    caffe_gpu_gemv(CblasTrans, T_*N_, 4*H_, Dtype(1.), pre_gate_diff,
        bias_multiplier_.gpu_data(), Dtype(1.),
        this->blobs_[2]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient w.r.t. bottom data
    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, I_, 4*H_, Dtype(1.),
        pre_gate_diff, weight_i, Dtype(0.), bottom[0]->mutable_gpu_diff());
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(LstmLayer);

}  // namespace caffe
