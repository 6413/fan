#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_window
#define loco_context

//#define loco_rectangle
#define loco_nv12
//#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

#include "cudaenc.h"

#define CUDA_DRVAPI_CALL( call )                                                                                                 \
    do                                                                                                                           \
    {                                                                                                                            \
        CUresult err__ = call;                                                                                                   \
        if (err__ != CUDA_SUCCESS)                                                                                               \
        {                                                                                                                        \
            const char *szErrName = NULL;                                                                                        \
            cuGetErrorName(err__, &szErrName);                                                                                   \
            std::ostringstream errorLog;                                                                                         \
            errorLog << "CUDA driver API error " << szErrName ;                                                                  \
            throw NVENCException::makeNVENCException(errorLog.str(), NV_ENC_ERR_GENERIC, __FUNCTION__, __FILE__, __LINE__);      \
        }                                                                                                                        \
    }                                                                                                                            \
    while (0)

NvEncoderCuda::NvEncoderCuda(CUcontext cuContext, uint32_t nWidth, uint32_t nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat,
  uint32_t nExtraOutputDelay, bool bMotionEstimationOnly) :
  NvEncoder(NV_ENC_DEVICE_TYPE_CUDA, cuContext, nWidth, nHeight, eBufferFormat, nExtraOutputDelay, bMotionEstimationOnly),
  m_cuContext(cuContext)
{
  if (!m_hEncoder)
  {
    NVENC_THROW_ERROR("Encoder Initialization failed", NV_ENC_ERR_INVALID_DEVICE);
  }

  if (!m_cuContext)
  {
    NVENC_THROW_ERROR("Invalid Cuda Context", NV_ENC_ERR_INVALID_DEVICE);
  }
}

NvEncoderCuda::~NvEncoderCuda()
{
  ReleaseCudaResources();
}

void NvEncoderCuda::AllocateInputBuffers(int32_t numInputBuffers)
{
  if (!IsHWEncoderInitialized())
  {
    NVENC_THROW_ERROR("Encoder intialization failed", NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
  }

  // for MEOnly mode we need to allocate seperate set of buffers for reference frame
  int numCount = m_bMotionEstimationOnly ? 2 : 1;

  for (int count = 0; count < numCount; count++)
  {
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(m_cuContext));
    std::vector<void*> inputFrames;
    for (int i = 0; i < numInputBuffers; i++)
    {
      CUdeviceptr pDeviceFrame;
      uint32_t chromaHeight = GetNumChromaPlanes(GetPixelFormat()) * GetChromaHeight(GetPixelFormat(), GetMaxEncodeHeight());
      if (GetPixelFormat() == NV_ENC_BUFFER_FORMAT_YV12 || GetPixelFormat() == NV_ENC_BUFFER_FORMAT_IYUV)
        chromaHeight = GetChromaHeight(GetPixelFormat(), GetMaxEncodeHeight());
      CUDA_DRVAPI_CALL(cuMemAllocPitch((CUdeviceptr*)&pDeviceFrame,
        &m_cudaPitch,
        GetWidthInBytes(GetPixelFormat(), GetMaxEncodeWidth()),
        GetMaxEncodeHeight() + chromaHeight, 16));
      inputFrames.push_back((void*)pDeviceFrame);
    }
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));

    RegisterResources(inputFrames,
      NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
      GetMaxEncodeWidth(),
      GetMaxEncodeHeight(),
      (int)m_cudaPitch,
      GetPixelFormat(),
      (count == 1) ? true : false);
  }
}

void NvEncoderCuda::ReleaseInputBuffers()
{
  ReleaseCudaResources();
}

void NvEncoderCuda::ReleaseCudaResources()
{
  if (!m_hEncoder)
  {
    return;
  }

  if (!m_cuContext)
  {
    return;
  }

  UnregisterResources();

  cuCtxPushCurrent(m_cuContext);

  for (uint32_t i = 0; i < m_vInputFrames.size(); ++i)
  {
    if (m_vInputFrames[i].inputPtr)
    {
      cuMemFree(reinterpret_cast<CUdeviceptr>(m_vInputFrames[i].inputPtr));
    }
  }
  m_vInputFrames.clear();

  for (uint32_t i = 0; i < m_vReferenceFrames.size(); ++i)
  {
    if (m_vReferenceFrames[i].inputPtr)
    {
      cuMemFree(reinterpret_cast<CUdeviceptr>(m_vReferenceFrames[i].inputPtr));
    }
  }
  m_vReferenceFrames.clear();

  cuCtxPopCurrent(NULL);
  m_cuContext = nullptr;
}

void NvEncoderCuda::CopyToDeviceFrame(CUcontext device,
  void* pSrcFrame,
  uint32_t nSrcPitch,
  CUdeviceptr pDstFrame,
  uint32_t dstPitch,
  int width,
  int height,
  CUmemorytype srcMemoryType,
  NV_ENC_BUFFER_FORMAT pixelFormat,
  const uint32_t dstChromaOffsets[],
  uint32_t numChromaPlanes,
  bool bUnAlignedDeviceCopy)
{
  if (srcMemoryType != CU_MEMORYTYPE_HOST && srcMemoryType != CU_MEMORYTYPE_DEVICE)
  {
    NVENC_THROW_ERROR("Invalid source memory type for copy", NV_ENC_ERR_INVALID_PARAM);
  }

  CUDA_DRVAPI_CALL(cuCtxPushCurrent(device));

  uint32_t srcPitch = nSrcPitch ? nSrcPitch : NvEncoder::GetWidthInBytes(pixelFormat, width);
  CUDA_MEMCPY2D m = { 0 };
  m.srcMemoryType = srcMemoryType;
  if (srcMemoryType == CU_MEMORYTYPE_HOST)
  {
    m.srcHost = pSrcFrame;
  }
  else
  {
    m.srcDevice = (CUdeviceptr)pSrcFrame;
  }
  m.srcPitch = srcPitch;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstDevice = pDstFrame;
  m.dstPitch = dstPitch;
  m.WidthInBytes = NvEncoder::GetWidthInBytes(pixelFormat, width);
  m.Height = height;
  if (bUnAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE)
  {
    CUDA_DRVAPI_CALL(cuMemcpy2DUnaligned(&m));
  }
  else
  {
    CUDA_DRVAPI_CALL(cuMemcpy2D(&m));
  }

  std::vector<uint32_t> srcChromaOffsets;
  NvEncoder::GetChromaSubPlaneOffsets(pixelFormat, srcPitch, height, srcChromaOffsets);
  uint32_t chromaHeight = NvEncoder::GetChromaHeight(pixelFormat, height);
  uint32_t destChromaPitch = NvEncoder::GetChromaPitch(pixelFormat, dstPitch);
  uint32_t srcChromaPitch = NvEncoder::GetChromaPitch(pixelFormat, srcPitch);
  uint32_t chromaWidthInBytes = NvEncoder::GetChromaWidthInBytes(pixelFormat, width);

  for (uint32_t i = 0; i < numChromaPlanes; ++i)
  {
    if (chromaHeight)
    {
      if (srcMemoryType == CU_MEMORYTYPE_HOST)
      {
        m.srcHost = ((uint8_t*)pSrcFrame + srcChromaOffsets[i]);
      }
      else
      {
        m.srcDevice = (CUdeviceptr)((uint8_t*)pSrcFrame + srcChromaOffsets[i]);
      }
      m.srcPitch = srcChromaPitch;

      m.dstDevice = (CUdeviceptr)((uint8_t*)pDstFrame + dstChromaOffsets[i]);
      m.dstPitch = destChromaPitch;
      m.WidthInBytes = chromaWidthInBytes;
      m.Height = chromaHeight;
      if (bUnAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE)
      {
        CUDA_DRVAPI_CALL(cuMemcpy2DUnaligned(&m));
      }
      else
      {
        CUDA_DRVAPI_CALL(cuMemcpy2D(&m));
      }
    }
  }
  CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
}

void NvEncoderCuda::CopyToDeviceFrame(CUcontext device,
  void* pSrcFrame,
  uint32_t nSrcPitch,
  CUdeviceptr pDstFrame,
  uint32_t dstPitch,
  int width,
  int height,
  CUmemorytype srcMemoryType,
  NV_ENC_BUFFER_FORMAT pixelFormat,
  CUdeviceptr dstChromaDevicePtrs[],
  uint32_t dstChromaPitch,
  uint32_t numChromaPlanes,
  bool bUnAlignedDeviceCopy)
{
  if (srcMemoryType != CU_MEMORYTYPE_HOST && srcMemoryType != CU_MEMORYTYPE_DEVICE)
  {
    NVENC_THROW_ERROR("Invalid source memory type for copy", NV_ENC_ERR_INVALID_PARAM);
  }

  CUDA_DRVAPI_CALL(cuCtxPushCurrent(device));

  uint32_t srcPitch = nSrcPitch ? nSrcPitch : NvEncoder::GetWidthInBytes(pixelFormat, width);
  CUDA_MEMCPY2D m = { 0 };
  m.srcMemoryType = srcMemoryType;
  if (srcMemoryType == CU_MEMORYTYPE_HOST)
  {
    m.srcHost = pSrcFrame;
  }
  else
  {
    m.srcDevice = (CUdeviceptr)pSrcFrame;
  }
  m.srcPitch = srcPitch;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstDevice = pDstFrame;
  m.dstPitch = dstPitch;
  m.WidthInBytes = NvEncoder::GetWidthInBytes(pixelFormat, width);
  m.Height = height;
  if (bUnAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE)
  {
    CUDA_DRVAPI_CALL(cuMemcpy2DUnaligned(&m));
  }
  else
  {
    CUDA_DRVAPI_CALL(cuMemcpy2D(&m));
  }

  std::vector<uint32_t> srcChromaOffsets;
  NvEncoder::GetChromaSubPlaneOffsets(pixelFormat, srcPitch, height, srcChromaOffsets);
  uint32_t chromaHeight = NvEncoder::GetChromaHeight(pixelFormat, height);
  uint32_t srcChromaPitch = NvEncoder::GetChromaPitch(pixelFormat, srcPitch);
  uint32_t chromaWidthInBytes = NvEncoder::GetChromaWidthInBytes(pixelFormat, width);

  for (uint32_t i = 0; i < numChromaPlanes; ++i)
  {
    if (chromaHeight)
    {
      if (srcMemoryType == CU_MEMORYTYPE_HOST)
      {
        m.srcHost = ((uint8_t*)pSrcFrame + srcChromaOffsets[i]);
      }
      else
      {
        m.srcDevice = (CUdeviceptr)((uint8_t*)pSrcFrame + srcChromaOffsets[i]);
      }
      m.srcPitch = srcChromaPitch;

      m.dstDevice = dstChromaDevicePtrs[i];
      m.dstPitch = dstChromaPitch;
      m.WidthInBytes = chromaWidthInBytes;
      m.Height = chromaHeight;
      if (bUnAlignedDeviceCopy && srcMemoryType == CU_MEMORYTYPE_DEVICE)
      {
        CUDA_DRVAPI_CALL(cuMemcpy2DUnaligned(&m));
      }
      else
      {
        CUDA_DRVAPI_CALL(cuMemcpy2D(&m));
      }
    }
  }
  CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
}

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {

    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_matrices(
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
    //fan::vec2 ratio = window_size / window_size.max();
    //std::swap(ratio.x, ratio.y);
    //matrices.set_ortho(
    //  ortho_x * ratio.x, 
    //  ortho_y * ratio.y
    //);
    viewport.set(loco.get_context(), 0, d.size, d.size);
      });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid[5];
};

pile_t* pile = new pile_t;

#include _FAN_PATH(video/nvdec.h)

int main() {

  CUdevice device = { 0 };
  CUcontext context = { 0 };

  cuInit(0);

  int device_count = 0;
  fan::cuda::check_error(cuDeviceGetCount(&device_count));

  fan::print_format("{} cuda device(s) found", device_count);

  fan::cuda::check_error(cuDeviceGet(&device, 0));

  char name[80] = { 0 };
  fan::cuda::check_error(cuDeviceGetName(name, sizeof(name), device));

  fan::print_format("cuda device: {}", name);

  fan::cuda::check_error(cuCtxCreate(&context, 0, device));

  fan::vec2 encode_size(2560, 1440);
  NV_ENC_BUFFER_FORMAT fmt = NV_ENC_BUFFER_FORMAT_ARGB;

  NvEncoderCuda enc(context, encode_size.x, encode_size.y, fmt);

  NV_ENC_INITIALIZE_PARAMS encInitParams = { 0 };
  /// NVENCODEAPI video encoding configuration parameters
  NV_ENC_CONFIG encConfig = { 0 };

  ZeroMemory(&encInitParams, sizeof(encInitParams));
  ZeroMemory(&encConfig, sizeof(encConfig));


  encInitParams.encodeConfig = &encConfig;
  encInitParams.encodeWidth = encode_size.x;
  encInitParams.encodeHeight = encode_size.y;
  encInitParams.maxEncodeWidth = encode_size.x;
  encInitParams.maxEncodeHeight = encode_size.y;
  encConfig.gopLength = 5;

  enc.CreateDefaultEncoderParams(&encInitParams, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_LOW_LATENCY_HP_GUID);
  enc.CreateEncoder(&encInitParams);


  //int data[200 * 200 * 4];
  //memset(data, 100, sizeof(data));

  fan::string str;
  fan::io::file::read("ss", &str);

  const NvEncInputFrame* frame = enc.GetNextInputFrame();
  fan::cuda::check_error(cudaMemcpy((void*)frame->inputPtr, str.data(), str.size(), cudaMemcpyHostToDevice));
  //memset(frame->inputPtr, 256, 1080);
  std::vector<std::vector<uint8_t>> packet;
  enc.EncodeFrame(packet);
  fan::io::file::write("encode", packet[0], std::ios_base::binary);
};