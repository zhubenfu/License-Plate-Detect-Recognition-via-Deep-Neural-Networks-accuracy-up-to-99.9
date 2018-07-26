#include "caffe/layers/warp_ctc_loss_layer.hpp"

#ifdef USE_WARP_CTC
#include <ctcpp.h>

#include <limits>
#include <sstream>

using namespace CTC;

namespace caffe {

template <typename Dtype>
void WarpCTCLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    cudaStream_t stream;
    CHECK_EQ(cudaStreamCreate(&stream), CUDA_SUCCESS);

    const Dtype* const activations = bottom[0]->gpu_data();
    Dtype* gradients = bottom[0]->mutable_gpu_diff();
    const int alphabet_size = C_;
    const int minibatch = N_;
    vector<Dtype> costs(N_);

	flat_labels_.clear();
	if (bottom.size() == 2) {//bottom[0]=activations, bottom[1] is labels, shape: Batchsize*seq len
		const Blob<Dtype>* label_seq_blob = bottom[1];
		const Dtype *label_seq_d = label_seq_blob->cpu_data();
		int label_len_per_batch = label_seq_blob->channels();
		
		for (int n = 0; n < N_; ++n)
		{
			int curlen = 0;
			for (int l = 0; l < label_len_per_batch; ++l)
			{
				int label = label_seq_d[n*label_len_per_batch + l];
				if (label == blank_index_)
					continue;
				flat_labels_.push_back(label);
				curlen++;
			}
			label_lengths_[n] = curlen;
			input_lengths_[n] = T_;
		}
	}
	else if(bottom.size() == 3) {
      ExtractInputData(bottom[1], bottom[2],
          &flat_labels_, &label_lengths_, &input_lengths_);
    } else if (bottom.size() == 4) {
      const Blob<Dtype>* seq_len_blob = bottom[1];
      const Blob<Dtype>* lab_len_blob = bottom[2];
      const Blob<Dtype>* label_seq_blob = bottom[3];

      const Dtype *seq_len_d = seq_len_blob->cpu_data();
      const Dtype *lab_len_d = lab_len_blob->cpu_data();
      const Dtype *label_seq_d = label_seq_blob->cpu_data();

      int accumulated = 0;
      CHECK_EQ(seq_len_blob->count(), lab_len_blob->count());
      for (int i = 0; i < seq_len_blob->count(); ++i) {
        label_lengths_[i] = lab_len_d[i];
        input_lengths_[i] = seq_len_d[i];
        accumulated += lab_len_d[i];
      }

      flat_labels_.clear();
      flat_labels_.reserve(accumulated);
      for (int n = 0; n < N_; ++n) {
        for (int t = 0; t < label_lengths_[n]; ++t) {
          flat_labels_.push_back(label_seq_d[label_seq_blob->offset(t, n)]);
        }
      }
    } else {
      LOG(FATAL) << "Unsupported blobs shape";
    }

    size_t workspace_alloc_bytes_;

    ctcOptions options;
    options.loc = CTC_GPU;
    options.blank_label = blank_index_;
    options.stream = stream;

    ctcStatus_t status = get_workspace_size<Dtype>(label_lengths_.data(),
                                            input_lengths_.data(),
                                            alphabet_size,
                                            minibatch,
                                            options,
                                            &workspace_alloc_bytes_);
    CHECK_EQ(status, CTC_STATUS_SUCCESS) << "CTC Error: " << ctcGetStatusString(status);

    if (!workspace_ || workspace_->size() < workspace_alloc_bytes_) {
      workspace_.reset(new SyncedMemory(workspace_alloc_bytes_));
    }

    status = compute_ctc_loss_gpu(activations,
                              gradients,
                              flat_labels_.data(),
                              label_lengths_.data(),
                              input_lengths_.data(),
                              alphabet_size,
                              minibatch,
                              costs.data(),
                              workspace_->mutable_gpu_data(),
                              options
                              );

    /*std::stringstream ss;
    for (int i = 0; i < flat_labels_.size(); ++i) {ss << flat_labels_[i] << " ";}
    ss << std::endl;
    for (int i = 0; i < label_lengths_.size(); ++i) {ss << label_lengths_[i] << " ";}
    ss << std::endl;
    for (int i = 0; i < input_lengths_.size(); ++i) {ss << input_lengths_[i] << " ";}
    ss << std::endl;
    LOG(INFO) << ss.str();*/


    CHECK_EQ(status, CTC_STATUS_SUCCESS) << "CTC Error: " << ctcGetStatusString(status);


    // output loss
    Dtype &loss = top[0]->mutable_cpu_data()[0];
    loss = 0;
    int num = 0;
    for (int n = 0; n < N_; ++n) {
      if (costs[n] < std::numeric_limits<Dtype>::infinity()) {
        loss += costs[n];
        ++num;
      }
    }

    loss /= num;

    CHECK_EQ(cudaStreamDestroy(stream), CUDA_SUCCESS);
}

template <typename Dtype>
void WarpCTCLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(propagate_down[0], true)
        << "Required to propagate to probabilities";
  if (propagate_down.size() >= 3)
  {
	  CHECK_EQ(propagate_down[1], false)
		  << "Cannot propagate to sequence indicators";
	  CHECK_EQ(propagate_down[2], false)
		  << "Cannot propagate to target label sequence";
  }
  else if (propagate_down.size() == 2)
  {
	  CHECK_EQ(propagate_down[1], false)
		  << "Cannot propagate to target label sequence";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WarpCTCLossLayer);

}  // namespace caffe

#endif  // USE_WARP_CTC
