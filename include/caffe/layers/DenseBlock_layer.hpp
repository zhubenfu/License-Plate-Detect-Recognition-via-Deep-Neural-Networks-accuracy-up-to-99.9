#ifndef CAFFE_DENSEBLOCK_LAYER_HPP_
#define CAFFE_DENSEBLOCK_LAYER_HPP_

#include <vector>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DenseBlockLayer : public Layer<Dtype> {
 public:
  explicit DenseBlockLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top); 
  
  virtual inline const char* type() const { return "DenseBlock"; } 

  virtual void Forward_cpu_public(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu_public(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void Forward_gpu_public(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  void Backward_gpu_public(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  virtual void syncBlobs(DenseBlockLayer<Dtype>* originLayer);

  virtual void setLogId(int uid);

  virtual void logInternal_cpu(string dir);

  void logInternal_gpu(string dir,int transitionIdx,bool logDynamic,bool logDiff);

  int initChannel, growthRate, numTransition; 
  int N,H,W; //N,H,W of the input tensor, inited in reshape phase
  
  bool useDropout;
  float dropoutAmount;
  unsigned long long DB_randomSeed;
  bool useBC;
  bool BC_ultra_spaceEfficient;
  
 protected:
  
  virtual void CPU_Initialization();

  void GPU_Initialization();
  void reshape_gpu_data(int oldh, int oldw, int oldn, int h, int w, int newn);

  virtual void LoopEndCleanup_cpu();

  void LoopEndCleanup_gpu();

  void resetDropoutDesc(); 

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  //start logging specific data: for debugging
  int logId;
  //end logging specific data

  //performance related parameter
  int gpu_idx_;

  //common Blobs for both CPU & GPU mode
  //in this->blobs_, containing all filters for Convolution, scalers and bias for BN
  
  //start CPU specific data section
  bool cpuInited;
  //at T has shape (1,initC+T*growth,1,1)
  vector<Blob<Dtype>*> batch_Mean; 
  vector<Blob<Dtype>*> batch_Var;
  vector<Blob<Dtype>*> batch_Mean4G;
  vector<Blob<Dtype>*> batch_Var4G;

  vector<Blob<Dtype>*> merged_conv;//at T has shape (N,initC+T*growth,H,W), but this vector has T+1 elements

  vector<Blob<Dtype>*> BN_XhatVec;//at T has shape (N,initC+T*growth,H,W)
  vector<Blob<Dtype>*> postBN_blobVec;
  vector<Blob<Dtype>*> postReLU_blobVec;
  vector<Blob<Dtype>*> postConv_blobVec;//at T has shape(N,growth,H,W)
  //BC related CPU 
  vector<Blob<Dtype>*> BC_BN_XhatVec;//at T has shape(N,4*growthRate,H,W)
  vector<Blob<Dtype>*> postBN_BCVec;
  vector<Blob<Dtype>*> postReLU_BCVec;
  vector<Blob<Dtype>*> postConv_BCVec; 
  //end CPU specific data section

  int trainCycleIdx; //used in BN train phase for EMA Mean/Var estimation
					 //convolution Related
  int pad_h, pad_w, conv_verticalStride, conv_horizentalStride;
  int filter_H, filter_W;
  //Decay value used in EMA of BN
  Dtype EMA_decay;

#ifndef CPU_ONLY
  //start GPU specific data section
  //GPU ptr for efficient space usage only, these pointers not allocated when CPU_ONLY, these are not Blobs because Descriptor is not traditional 
  //bool gpuInited;
  Dtype* postConv_data_gpu;
  Dtype* postConv_grad_gpu;
  Dtype* postDropout_data_gpu;
  Dtype* postDropout_grad_gpu;
  Dtype* postBN_data_gpu;
  Dtype* postBN_grad_gpu;
  Dtype* postReLU_data_gpu;
  Dtype* postReLU_grad_gpu;
  Dtype* workspace;
  Dtype* workspace2;
  //gpu workspace size
  int workspace_size_bytes;

  vector<Dtype*> ResultSaveMean_gpu;
  vector<Dtype*> ResultSaveInvVariance_gpu;
  vector<void*> dropout_state_gpu;
  vector<size_t> dropout_stateSize;
  vector<void*> dropout_reserve_gpu;
  vector<size_t> dropout_reserveSize;
  Dtype* Mean_tmp;//used in BN inf
  Dtype* Var_tmp;//used in BN inf
  
  //BC related parameters 
  vector<Dtype*> postConv_4GVec; //used if not ultra space efficient mode
  Dtype* postConv_4G; //used if ultra space efficient mode
  Dtype* postBN_4G;
  Dtype* postReLU_4G;  
  Dtype* postConv_4G_grad;
  Dtype* postBN_4G_grad;
  Dtype* postReLU_4G_grad;
  cudnnTensorDescriptor_t * quadG_tensorDesc;
  cudnnTensorDescriptor_t * quadG_paramDesc;
  cudnnConvolutionDescriptor_t* convBC_Descriptor;
   vector<Dtype*> BC_MeanInfVec;
  vector<Dtype*> BC_VarInfVec;
  vector<Dtype*> ResultSaveMean_BC;
  vector<Dtype*> ResultSaveInvVariance_BC;
   vector<cudnnFilterDescriptor_t *> BC_filterDescriptorVec;
  //chosen Fwd,BwdFilter,BwdData algos for BC-Conv/Normal-Conv
  vector<cudnnConvolutionFwdAlgo_t *> conv_FwdAlgoVec;
  vector<cudnnConvolutionFwdAlgo_t *> BC_FwdAlgoVec;
  vector<cudnnConvolutionBwdFilterAlgo_t *> conv_BwdFilterAlgoVec;
  vector<cudnnConvolutionBwdFilterAlgo_t *> BC_BwdFilterAlgoVec;
  vector<cudnnConvolutionBwdDataAlgo_t *> conv_BwdDataAlgoVec;
  vector<cudnnConvolutionBwdDataAlgo_t *> BC_BwdDataAlgoVec; 
   //BC_dropout
  //vector<void*> BC_dropout_state;
  //vector<void*> BC_dropout_reserve;
  //vector<size_t> BC_dropout_stateSize;
  //vector<size_t> BC_dropout_reserveSize;
  //Dtype* postDropout_4G;
  //Dtype* postDropout_4G_grad;
  
 
   //gpu handles and descriptors
  cudnnHandle_t* cudnnHandlePtr;
  cudaStream_t* cudaPrimalStream;
  vector<cudnnHandle_t*> extraHandles;
  vector<cudaStream_t*> extraStreams;

  vector<cudnnTensorDescriptor_t *> tensorDescriptorVec_conv_x;//local Conv X
  cudnnTensorDescriptor_t * tensorDescriptor_conv_y;//local Conv Y
  vector<cudnnTensorDescriptor_t *> tensorDescriptor_BN;//<channelwise>
  //Dropout descriptor 
  vector<cudnnDropoutDescriptor_t *> dropoutDescriptorVec;
  //filter descriptor for conv
  vector<cudnnFilterDescriptor_t *> filterDescriptorVec;
  //ReLU Activation Descriptor  
  cudnnActivationDescriptor_t* ReLUDesc;
  //conv descriptor for conv
  cudnnConvolutionDescriptor_t* conv_Descriptor;
#endif
  //end GPU specific data setion
};

}  // namespace caffe

#endif  // CAFFE_DENSEBLOCK_LAYER_HPP_

