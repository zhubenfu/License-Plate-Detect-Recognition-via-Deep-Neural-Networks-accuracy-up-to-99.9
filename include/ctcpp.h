/** \file ctcpp.h
 * Contains a simple CPP interface to call fast CPU and GPU based computation
 * of the CTC loss.
 */

#pragma once

#include "ctc.h"

namespace CTC {

/** Returns a single integer which specifies the API version of the warpctc library */
int get_warpctc_version();

/** Returns a string containing a description of status that was passed in
*  \param[in] status identifies which string should be returned
*  \return C style string containing the text description
*  */
const char* ctcGetStatusString(ctcStatus_t status);

/** Compute the connectionist temporal classification loss between a sequence
 *  of probabilities and a ground truth labeling.  Optionally compute the
 *  gradient with respect to the inputs.
 * \param [in] activations pointer to the activations in either CPU or GPU
 *             addressable memory, depending on info.  We assume a fixed
 *             memory layout for this 3 dimensional tensor, which has dimension
 *             (t, n, p), where t is the time index, n is the minibatch index,
 *             and p indexes over probabilities of each symbol in the alphabet.
 *             The memory layout is (t, n, p) in C order (slowest to fastest changing
 *             index, aka row-major), or (p, n, t) in Fortran order (fastest to slowest
 *             changing index, aka column-major). We also assume strides are equal to
 *             dimensions - there is no padding between dimensions.
 *             More precisely, element (t, n, p), for a problem with mini_batch examples
 *             in the mini batch, and alphabet_size symbols in the alphabet, is located at:
 *             activations[(t * mini_batch + n) * alphabet_size + p]
 * \param [out] gradients if not NULL, then gradients are computed.  Should be
 *              allocated in the same memory space as probs and memory
 *              ordering is identical.
 * \param [in]  flat_labels Always in CPU memory.  A concatenation
 *              of all the labels for the minibatch.
 * \param [in]  label_lengths Always in CPU memory. The length of each label
 *              for each example in the minibatch.
 * \param [in]  input_lengths Always in CPU memory.  The number of time steps
 *              for each sequence in the minibatch.
 * \param [in]  alphabet_size The number of possible output symbols.  There
 *              should be this many probabilities for each time step.
 * \param [in]  mini_batch How many examples in a minibatch.
 * \param [out] costs Always in CPU memory.  The cost of each example in the
 *              minibatch.
 * \param [in,out] workspace In same memory space as probs. Should be of
 *                 size requested by get_workspace_size.
 * \param [in]  options see struct ctcOptions
 *
 *  \return Status information
 *
 * */
template<typename Dtype>
ctcStatus_t compute_ctc_loss_cpu(const Dtype* const activations,
                             Dtype* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             Dtype *costs,
                             void *workspace,
                             ctcOptions options);

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
	ctcOptions options);


/** For a given set of labels and minibatch size return the required workspace
 *  size.  This will need to be allocated in the same memory space as your
 *  probabilities.
 * \param [in]  label_lengths Always in CPU memory. The length of each label
 *              for each example in the minibatch.
 * \param [in]  input_lengths Always in CPU memory.  The number of time steps
 *              for each sequence in the minibatch.
 * \param [in]  alphabet_size How many symbols in the alphabet or, equivalently,
 *              the number of probabilities at each time step
 * \param [in]  mini_batch How many examples in a minibatch.
 * \param [in]  info see struct ctcOptions
 * \param [out] size_bytes is pointer to a scalar where the memory
 *              requirement in bytes will be placed. This memory should be allocated
 *              at the same place, CPU or GPU, that the probs are in
 *
 *  \return Status information
 **/
template<typename Dtype>
ctcStatus_t get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size, int minibatch,
                               ctcOptions info,
                               size_t* size_bytes);



} // namespace ctc
