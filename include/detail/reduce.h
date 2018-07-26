#pragma once

template<typename ProbT>
ctcStatus_t reduce_negate(const ProbT* input, ProbT* output, int rows, int cols, bool axis, cudaStream_t stream);

template<typename ProbT>
ctcStatus_t reduce_exp(const ProbT* input, ProbT* output, int rows, int cols, bool axis, cudaStream_t stream);

template<typename ProbT>
ctcStatus_t reduce_max(const ProbT* input, ProbT* output, int rows, int cols, bool axis, cudaStream_t stream);
