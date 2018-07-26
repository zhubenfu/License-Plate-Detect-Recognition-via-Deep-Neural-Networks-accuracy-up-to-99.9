
//forward declare of CUDA typedef to avoid needing to pull in CUDA headers
#pragma once

typedef struct CUstream_st* CUstream;

typedef enum {
    CTC_STATUS_SUCCESS = 0,
    CTC_STATUS_MEMOPS_FAILED = 1,
    CTC_STATUS_INVALID_VALUE = 2,
    CTC_STATUS_EXECUTION_FAILED = 3,
    CTC_STATUS_UNKNOWN_ERROR = 4
} ctcStatus_t;



typedef enum {
    CTC_CPU = 0,
    CTC_GPU = 1
} ctcComputeLocation;

/** Structure used for options to the CTC compution.  Applications
 *  should zero out the array using memset and sizeof(struct
 *  ctcOptions) in C or default initialization (e.g. 'ctcOptions
 *  options{};' or 'auto options = ctcOptions{}') in C++ to ensure
 *  forward compatibility with added options. */
struct ctcOptions {
    /// indicates where the ctc calculation should take place {CTC_CPU | CTC_GPU}
    ctcComputeLocation loc;
    union {
        /// used when loc == CTC_CPU, the maximum number of threads that can be used
        unsigned int num_threads;

        /// used when loc == CTC_GPU, which stream the kernels should be launched in
        CUstream stream;
    };

    /// the label value/index that the CTC calculation should use as the blank label
    int blank_label;
};