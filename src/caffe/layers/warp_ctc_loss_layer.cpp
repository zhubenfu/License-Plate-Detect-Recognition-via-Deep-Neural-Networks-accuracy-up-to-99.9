#include "caffe/layers/warp_ctc_loss_layer.hpp"

#ifdef USE_WARP_CTC
#include <ctcpp.h>

#include <limits>

using namespace CTC;

namespace caffe {

template <typename Dtype>
WarpCTCLossLayer<Dtype>::WarpCTCLossLayer(const LayerParameter& param)
     : LossLayer<Dtype>(param),
       T_(0),
       N_(0),
       C_(0) {
  blank_index_ = param.ctc_loss_param().blank_index();
}

template <typename Dtype>
WarpCTCLossLayer<Dtype>::~WarpCTCLossLayer() {
}

template <typename Dtype>
void WarpCTCLossLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  const Blob<Dtype>* probs = bottom[0];//TxNxC


  T_ = probs->num();
  N_ = probs->channels();
  C_ = probs->height();
  CHECK_EQ(probs->width(), 1);

  if (bottom.size() == 3) {
    const Blob<Dtype>* seq_ind = bottom[1];
    const Blob<Dtype>* label_seq = bottom[2];
    CHECK_EQ(T_, seq_ind->num());
    CHECK_EQ(N_, seq_ind->channels());
    CHECK_EQ(N_, label_seq->channels());
  } else if (bottom.size() == 4) {
    const Blob<Dtype>* seq_len_blob = bottom[1];
    const Blob<Dtype>* lab_len_blob = bottom[2];
    const Blob<Dtype>* label_seq_blob = bottom[3];
    CHECK_EQ(N_, seq_len_blob->count());
    CHECK_EQ(N_, lab_len_blob->count());
    CHECK_EQ(N_, label_seq_blob->channels());
  }
  else if (bottom.size() == 2)//input seq + labels
  {
	  const Blob<Dtype>* label_seq = bottom[1];
	  CHECK_EQ(N_, label_seq->num());
  }
  else {
    LOG(FATAL) << "Unsupported blobs shape";
  }

  label_lengths_.resize(N_);
  input_lengths_.resize(N_);
}

template <typename Dtype>
void WarpCTCLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void WarpCTCLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    const Dtype* const activations = bottom[0]->cpu_data();
    Dtype* gradients = bottom[0]->mutable_cpu_diff();
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
				if(label == blank_index_)
					continue;
				flat_labels_.push_back(label);
				curlen++;
			}
			label_lengths_[n] = curlen;
			input_lengths_[n] = T_;
		}
	}
    else if (bottom.size() == 3) {
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

	//remove repeat blank labels


    size_t workspace_alloc_bytes_;

    ctcOptions options;
    options.loc = CTC_CPU;
    options.num_threads = 8;
    options.blank_label = blank_index_;

    ctcStatus_t status = get_workspace_size<Dtype>(label_lengths_.data(),
                                            input_lengths_.data(),
                                            alphabet_size,
                                            minibatch,
                                            options,
                                            &workspace_alloc_bytes_);
    CHECK_EQ(status, CTC_STATUS_SUCCESS) << "CTC Error: " << ctcGetStatusString(status);

    if (!workspace_ || workspace_->size() < workspace_alloc_bytes_) {
      workspace_.reset(new SyncedMemory(workspace_alloc_bytes_ * sizeof(char)));
    }

    status = compute_ctc_loss_cpu(activations,
                              gradients,
                              flat_labels_.data(),
                              label_lengths_.data(),
                              input_lengths_.data(),
                              alphabet_size,
                              minibatch,
                              costs.data(),
                              workspace_->mutable_cpu_data(),
                              options
                              );

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

	int gcnt = bottom[0]->count();
	Dtype sumg = 0;
	for (int i=0;i<gcnt;i++)
	{
		sumg += fabs(gradients[i]);
	}
	//LOG(INFO) << "mean ctc loss=" << loss << ",N_="<<N_<<",num="<<num << ", mean gradients="<<sumg/gcnt;
}

template <typename Dtype>
void WarpCTCLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

template <typename Dtype>
void WarpCTCLossLayer<Dtype>::ExtractInputData(const Blob<Dtype>* seq_ind_blob,
                                           const Blob<Dtype>* labels_blob,
                                           vector<int>* flat_labels,
                                           vector<int>* label_lengths,
                                           vector<int>* input_lengths) {

  const Dtype* seq_ind = CHECK_NOTNULL(seq_ind_blob)->cpu_data();
  const Dtype* target_seq = CHECK_NOTNULL(labels_blob)->cpu_data();

  CHECK_NOTNULL(flat_labels)->clear();
  flat_labels->reserve(T_ * N_);  // maximum required
  CHECK_NOTNULL(label_lengths)->resize(N_);
  CHECK_NOTNULL(input_lengths)->resize(N_);

  // compute the sequence length and label length
  int* seq_len = input_lengths->data();
  int* label_len = label_lengths->data();

  int label_offset = 0;
  //if (blank_index_ == -1) {
  if (blank_index_ == 0) {//modified by jxs
    label_offset = 1;
  }

  for (int n = 0; n < N_; ++n) {
    seq_len[n] = T_;  // default value is maximal allowed length
    label_len[n] = T_;  // default value is maximal allowed length

    const Dtype *seq = seq_ind + n;
    const Dtype *label = target_seq + n;

    // sequence indicators start with seq == 0.0 to indicate the start of a
    // sequence. Skip at t = 0, so start at t = 1
    seq += N_;
    for (int t = 1; t < T_; ++t) {
      if (static_cast<int>(*seq + 0.5) == 0) {
        seq_len[n] = t;
        break;
      }
      seq += N_;
    }

    // label indicators are negative if the sequence has ended
    for (int t = 0; t < T_; ++t) {
      if (*label < 0.0) {
        label_len[n] = t;
        break;
      }
      // Note that the blank label will be 0
      flat_labels->push_back(static_cast<int>(*label + 0.5) + label_offset);
      label += N_;
    }

    // if the label length is 0, the seq_len is 1 (0 following 0)
    // set seq_len to 0 in this case aswell, to skip this example
    if (label_len[n] == 0) {
        CHECK_LE(seq_len[n], 1);
        seq_len[n] = 0;
    }

    CHECK_LE(label_len[n], seq_len[n])
         << "The label length must be smaller or equals the sequence length!";
  }
}

#ifdef CPU_ONLY
STUB_GPU(WarpCTCLossLayer);
#endif

INSTANTIATE_CLASS(WarpCTCLossLayer);
REGISTER_LAYER_CLASS(WarpCTCLoss);

}  // namespace caffe

#endif  // USE_WARP_CTC
