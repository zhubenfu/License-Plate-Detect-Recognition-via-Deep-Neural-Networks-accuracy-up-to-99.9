#include <cstddef>
#include <iostream>
#include <algorithm>

#include <ctcpp.h>

#include "detail/gpu_ctc.h"

namespace CTC {

	template<typename Dtype>
	ctcStatus_t compute_ctc_loss_gpu(const Dtype* const activations,
		Dtype* gradients,
		const int* const flat_labels,
		const int* const label_lengths,
		const int* const input_lengths,
		int alphabet_size,
		int minibatch,
		Dtype *costs,
		void *workspace,
		ctcOptions options) {

		if (activations == nullptr ||
			flat_labels == nullptr ||
			label_lengths == nullptr ||
			input_lengths == nullptr ||
			costs == nullptr ||
			workspace == nullptr ||
			alphabet_size <= 0 ||
			minibatch <= 0)
			return CTC_STATUS_INVALID_VALUE;

		GpuCTC<Dtype> ctc(alphabet_size, minibatch, workspace, options.stream,
			options.blank_label);

		if (gradients != NULL)
			return ctc.cost_and_grad(activations, gradients, costs,
				flat_labels, label_lengths,
				input_lengths);
		else
			return ctc.score_forward(activations, costs, flat_labels,
				label_lengths, input_lengths);

	}


	template <typename Dtype>
	ctcStatus_t get_workspace_size_gpu(const int* const label_lengths,
		const int* const input_lengths,
		int alphabet_size, int minibatch,
		ctcOptions options,
		size_t* size_bytes)
	{
		if (label_lengths == nullptr ||
			input_lengths == nullptr ||
			size_bytes == nullptr ||
			alphabet_size <= 0 ||
			minibatch <= 0)
			return CTC_STATUS_INVALID_VALUE;

		// This is the max of all S and T for all examples in the minibatch.
		int maxL = *std::max_element(label_lengths, label_lengths + minibatch);
		int maxT = *std::max_element(input_lengths, input_lengths + minibatch);

		const int S = 2 * maxL + 1;

		*size_bytes = 0;

		// GPU storage
		//nll_forward, nll_backward
		*size_bytes += 2 * sizeof(Dtype) * minibatch;

		//repeats
		*size_bytes += sizeof(int) * minibatch;

		//label offsets
		*size_bytes += sizeof(int) * minibatch;

		//utt_length
		*size_bytes += sizeof(int) * minibatch;

		//label lengths
		*size_bytes += sizeof(int) * minibatch;

		//labels without blanks - overallocate for now
		*size_bytes += sizeof(int) * maxL * minibatch;

		//labels with blanks
		*size_bytes += sizeof(int) * S * minibatch;

		//alphas
		*size_bytes += sizeof(Dtype) * S * maxT * minibatch;

		//denoms
		*size_bytes += sizeof(Dtype) * maxT * minibatch;

		//probs (since we will pass in activations)
		*size_bytes += sizeof(Dtype) * alphabet_size * maxT * minibatch;


		return CTC_STATUS_SUCCESS;
	}

	template
		ctcStatus_t compute_ctc_loss_gpu<float>(const float* const activations,
			float* gradients,
			const int* const flat_labels,
			const int* const label_lengths,
			const int* const input_lengths,
			int alphabet_size,
			int minibatch,
			float *costs,
			void *workspace,
			ctcOptions options);


	template
		ctcStatus_t compute_ctc_loss_gpu<double>(const double* const activations,
			double* gradients,
			const int* const flat_labels,
			const int* const label_lengths,
			const int* const input_lengths,
			int alphabet_size,
			int minibatch,
			double *costs,
			void *workspace,
			ctcOptions);


	template
		ctcStatus_t get_workspace_size_gpu<float>(const int* const label_lengths,
			const int* const input_lengths,
			int alphabet_size, int minibatch,
			ctcOptions,
			size_t* size_bytes);


	template
		ctcStatus_t get_workspace_size_gpu<double>(const int* const label_lengths,
			const int* const input_lengths,
			int alphabet_size, int minibatch,
			ctcOptions,
			size_t* size_bytes);
}  // namespace ctc

