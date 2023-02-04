#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 1
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan


#define loco_window
#define loco_context

//#define loco_rectangle
#define loco_nv12
//#define loco_sprite
#include _FAN_PATH(graphics/loco.h)

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
loco_t::nv12_t::properties_t p;

#include _FAN_PATH(video/nvdec.h)


#include <nvEncodeAPI.h>

#define CHECK_CUDA_ERROR(err) { if (err != cudaSuccess) { std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); } }
#define CHECK_NVENC_ERROR(err) { if (err != NV_ENC_SUCCESS) { std::cout << "NVENC error: " << err << std::endl; exit(1); } }

const int width = 1920;
const int height = 1080;
const int frame_size = width * height * 4;
const int fps = 30;
const int bitrate = 1000000;

static constexpr auto hardcode_format = NV_ENC_BUFFER_FORMAT_ARGB;

// Encoder session handle
NV_ENCODE_API_FUNCTION_LIST func_list;

void* hEncoder = NULL;

// RGBA input image
unsigned char* rgba = new unsigned char[frame_size];

// H264 output stream
std::ofstream output;

CUcontext context = { 0 };
CUdevice device = { 0 };

size_t m_cudaPitch = 0;


struct NvEncInputFrame
{
  void* inputPtr = nullptr;
  uint32_t chromaOffsets[2];
  uint32_t numChromaPlanes;
  uint32_t pitch;
  uint32_t chromaPitch;
  NV_ENC_BUFFER_FORMAT bufferFormat;
  NV_ENC_INPUT_RESOURCE_TYPE resourceType;
};

std::vector<NvEncInputFrame> m_vInputFrames;
std::vector<NV_ENC_REGISTERED_PTR> m_vRegisteredResources;
std::vector<NvEncInputFrame> m_vReferenceFrames;
std::vector<NV_ENC_REGISTERED_PTR> m_vRegisteredResourcesForReference;


std::vector<NV_ENC_INPUT_PTR> m_vMappedInputBuffers;
std::vector<NV_ENC_INPUT_PTR> m_vMappedRefBuffers;
std::vector<NV_ENC_OUTPUT_PTR> m_vBitstreamOutputBuffer;
std::vector<NV_ENC_OUTPUT_PTR> m_vMVDataOutputBuffer;


std::vector<void*> m_vpCompletionEvent;

// Enco

uint32_t GetNumChromaPlanes(const NV_ENC_BUFFER_FORMAT bufferFormat)
{
  switch (bufferFormat)
  {
  case NV_ENC_BUFFER_FORMAT_NV12:
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
    return 1;
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
  case NV_ENC_BUFFER_FORMAT_YUV444:
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    return 2;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return 0;
  default:
    fan::throw_error("invalid format");
    return -1;
  }
}

uint32_t GetChromaHeight(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t lumaHeight)
{
  switch (bufferFormat)
  {
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
  case NV_ENC_BUFFER_FORMAT_NV12:
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
    return (lumaHeight + 1) / 2;
  case NV_ENC_BUFFER_FORMAT_YUV444:
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    return lumaHeight;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return 0;
  default:
    fan::throw_error("invalid format");
    return 0;
  }
}

uint32_t GetWidthInBytes(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t width)
{
  switch (bufferFormat) {
  case NV_ENC_BUFFER_FORMAT_NV12:
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
  case NV_ENC_BUFFER_FORMAT_YUV444:
    return width;
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    return width * 2;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return width * 4;
  default:
    fan::throw_error("invalid format");
    return 0;
  }
}


uint32_t GetChromaPitch(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t lumaPitch)
{
  switch (bufferFormat)
  {
  case NV_ENC_BUFFER_FORMAT_NV12:
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
  case NV_ENC_BUFFER_FORMAT_YUV444:
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    return lumaPitch;
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
    return (lumaPitch + 1) / 2;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return 0;
  default:
    fan::throw_error("invalid format");
    return -1;
  }
}

void GetChromaSubPlaneOffsets(const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t pitch, const uint32_t height, std::vector<uint32_t>& chromaOffsets)
{
  chromaOffsets.clear();
  switch (bufferFormat)
  {
  case NV_ENC_BUFFER_FORMAT_NV12:
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
    chromaOffsets.push_back(pitch * height);
    return;
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
    chromaOffsets.push_back(pitch * height);
    chromaOffsets.push_back(chromaOffsets[0] + (GetChromaPitch(bufferFormat, pitch) * GetChromaHeight(bufferFormat, height)));
    return;
  case NV_ENC_BUFFER_FORMAT_YUV444:
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    chromaOffsets.push_back(pitch * height);
    chromaOffsets.push_back(chromaOffsets[0] + (pitch * height));
    return;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return;
  default:
    fan::throw_error("invalid format");
    return;
  }
}

void RegisterResources(std::vector<void*> inputframes, NV_ENC_INPUT_RESOURCE_TYPE eResourceType,
  int width, int height, int pitch, NV_ENC_BUFFER_FORMAT bufferFormat, bool bReferenceFrame)
{
  for (uint32_t i = 0; i < inputframes.size(); ++i)
  {
    NV_ENC_REGISTER_RESOURCE registerResource = { NV_ENC_REGISTER_RESOURCE_VER };
    registerResource.resourceType = eResourceType;
    registerResource.resourceToRegister = (void*)inputframes[i];
    registerResource.width = width;
    registerResource.height = height;
    registerResource.pitch = pitch;
    registerResource.bufferFormat = bufferFormat;
    fan::cuda::check_error(func_list.nvEncRegisterResource(hEncoder, &registerResource));

    std::vector<uint32_t> _chromaOffsets;
    GetChromaSubPlaneOffsets(bufferFormat, pitch, height, _chromaOffsets);
    NvEncInputFrame inputframe = {};
    inputframe.inputPtr = (void*)inputframes[i];
    inputframe.chromaOffsets[0] = 0;
    inputframe.chromaOffsets[1] = 0;
    for (uint32_t ch = 0; ch < _chromaOffsets.size(); ch++)
    {
      inputframe.chromaOffsets[ch] = _chromaOffsets[ch];
    }
    inputframe.numChromaPlanes = GetNumChromaPlanes(bufferFormat);
    inputframe.pitch = pitch;
    inputframe.chromaPitch = GetChromaPitch(bufferFormat, pitch);
    inputframe.bufferFormat = bufferFormat;
    inputframe.resourceType = eResourceType;

    if (bReferenceFrame)
    {
      m_vRegisteredResourcesForReference.push_back(registerResource.registeredResource);
      m_vReferenceFrames.push_back(inputframe);
    }
    else
    {
      m_vRegisteredResources.push_back(registerResource.registeredResource);
      m_vInputFrames.push_back(inputframe);
    }
  }
}

void AllocateInputBuffers(int32_t numInputBuffers)
{

  // for MEOnly mode we need to allocate seperate set of buffers for reference frame
  int numCount = 1;

  for (int count = 0; count < numCount; count++)
  {
    std::vector<void*> inputFrames;
    for (int i = 0; i < numInputBuffers; i++)
    {
      CUdeviceptr pDeviceFrame;
      uint32_t chromaHeight = GetNumChromaPlanes(hardcode_format) * GetChromaHeight(hardcode_format, height);
      fan::cuda::check_error(cuMemAllocPitch((CUdeviceptr*)&pDeviceFrame,
        &m_cudaPitch,
        GetWidthInBytes(hardcode_format, width),
        height + chromaHeight, 16));
      inputFrames.push_back((void*)pDeviceFrame);
    }

    RegisterResources(inputFrames,
      NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
      width,
      height,
      (int)m_cudaPitch,
      hardcode_format,
      (count == 1) ? true : false);
  }
}

void CreateDefaultEncoderParams(NV_ENC_INITIALIZE_PARAMS* pIntializeParams, GUID codecGuid, GUID presetGuid) {


  memset(pIntializeParams->encodeConfig, 0, sizeof(NV_ENC_CONFIG));
  auto pEncodeConfig = pIntializeParams->encodeConfig;
  memset(pIntializeParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));
  pIntializeParams->encodeConfig = pEncodeConfig;


  pIntializeParams->encodeConfig->version = NV_ENC_CONFIG_VER;
  pIntializeParams->version = NV_ENC_INITIALIZE_PARAMS_VER;

  pIntializeParams->encodeGUID = codecGuid;
  pIntializeParams->presetGUID = presetGuid;
  pIntializeParams->encodeWidth = width;
  pIntializeParams->encodeHeight = height;
  pIntializeParams->darWidth = width;
  pIntializeParams->darHeight = height;
  pIntializeParams->frameRateNum = 30;
  pIntializeParams->frameRateDen = 1;
  pIntializeParams->enablePTD = 1;
  pIntializeParams->reportSliceOffsets = 0;
  pIntializeParams->enableSubFrameWrite = 0;
  pIntializeParams->maxEncodeWidth = width;
  pIntializeParams->maxEncodeHeight = height;
  //pIntializeParams->enableMEOnlyMode = m_bMotionEstimationOnly;

  NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER, { NV_ENC_CONFIG_VER } };
  fan::cuda::check_error(func_list.nvEncGetEncodePresetConfig(hEncoder, codecGuid, presetGuid, &presetConfig));
  memcpy(pIntializeParams->encodeConfig, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));
  pIntializeParams->encodeConfig->frameIntervalP = 1;
  pIntializeParams->encodeConfig->gopLength = NVENC_INFINITE_GOPLENGTH;

  pIntializeParams->encodeConfig->rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;

  if (pIntializeParams->presetGUID != NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID
    && pIntializeParams->presetGUID != NV_ENC_PRESET_LOSSLESS_HP_GUID)
  {
    pIntializeParams->encodeConfig->rcParams.constQP = { 28, 31, 25 };
  }

  if (pIntializeParams->encodeGUID == NV_ENC_CODEC_H264_GUID)
  {
    pIntializeParams->encodeConfig->encodeCodecConfig.h264Config.idrPeriod = pIntializeParams->encodeConfig->gopLength;
  }
  else if (pIntializeParams->encodeGUID == NV_ENC_CODEC_HEVC_GUID)
  {
    pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 = 0;
    pIntializeParams->encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = pIntializeParams->encodeConfig->gopLength;
  }

  return;
}

uint32_t m_iToSend = 0;
int32_t m_iGot = 0;

void GetEncodedPacket(std::vector<NV_ENC_OUTPUT_PTR>& vOutputBuffer, std::vector<std::vector<uint8_t>>& vPacket, bool bOutputDelay)
{
  unsigned i = 0;
  int iEnd = m_iToSend;
  for (; m_iGot < iEnd; m_iGot++)
  {
    NV_ENC_LOCK_BITSTREAM lockBitstreamData = { NV_ENC_LOCK_BITSTREAM_VER };
    lockBitstreamData.outputBitstream = vOutputBuffer[m_iGot % 1];
    lockBitstreamData.doNotWait = false;
    fan::cuda::check_error(func_list.nvEncLockBitstream(hEncoder, &lockBitstreamData));

    uint8_t* pData = (uint8_t*)lockBitstreamData.bitstreamBufferPtr;
    if (vPacket.size() < i + 1)
    {
      vPacket.push_back(std::vector<uint8_t>());
    }
    vPacket[i].clear();
    vPacket[i].insert(vPacket[i].end(), &pData[0], &pData[lockBitstreamData.bitstreamSizeInBytes]);
    i++;

    fan::cuda::check_error(func_list.nvEncUnlockBitstream(hEncoder, lockBitstreamData.outputBitstream));

    if (m_vMappedInputBuffers[m_iGot % 1])
    {
      fan::cuda::check_error(func_list.nvEncUnmapInputResource(hEncoder, m_vMappedInputBuffers[m_iGot % 1]));
      m_vMappedInputBuffers[m_iGot % 1] = nullptr;
    }

    //if (m_bMotionEstimationOnly && m_vMappedRefBuffers[m_iGot % 1])
    //{
    //  fan::cuda::check_error(func_list.nvEncUnmapInputResource(hEncoder, m_vMappedRefBuffers[m_iGot % 1]));
    //  m_vMappedRefBuffers[m_iGot % 1] = nullptr;
    //}
  }
}

void DoEncode(NV_ENC_INPUT_PTR inputBuffer, std::vector<std::vector<uint8_t>>& vPacket, NV_ENC_PIC_PARAMS* pPicParams)
{
  NV_ENC_PIC_PARAMS picParams = {};
  if (pPicParams)
  {
    picParams = *pPicParams;
  }
  picParams.version = NV_ENC_PIC_PARAMS_VER;
  picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  picParams.inputBuffer = inputBuffer;
  picParams.bufferFmt = hardcode_format;
  picParams.inputWidth = width;
  picParams.inputHeight = height;
  picParams.outputBitstream = m_vBitstreamOutputBuffer[m_iToSend % 1];
  picParams.completionEvent = m_vpCompletionEvent[m_iToSend % 1];
  NVENCSTATUS nvStatus = func_list.nvEncEncodePicture(hEncoder, &picParams);
  if (nvStatus == NV_ENC_SUCCESS || nvStatus == NV_ENC_ERR_NEED_MORE_INPUT)
  {
    m_iToSend++;
    GetEncodedPacket(m_vBitstreamOutputBuffer, vPacket, true);
  }
  else
  {
    fan::throw_error("");
    //NVENC_THROW_ERROR("nvEncEncodePicture API failed", nvStatus);
  }
}


int main() {
  // CUDA initialization
  cuInit(0);
  CHECK_CUDA_ERROR(cudaGetLastError());

  func_list = { NV_ENCODE_API_FUNCTION_LIST_VER };

  // NVENC initialization
  fan::cuda::check_error(NvEncodeAPICreateInstance(&func_list));

  int device_count = 0;
  fan::cuda::check_error(cuDeviceGetCount(&device_count));

  fan::print_format("{} cuda device(s) found", device_count);

  fan::cuda::check_error(cuDeviceGet(&device, 0));

  char name[80] = { 0 };
  fan::cuda::check_error(cuDeviceGetName(name, sizeof(name), device));

  fan::print_format("cuda device: {}", name);

  fan::cuda::check_error(cuCtxCreate(&context, 0, device));


  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
  encodeSessionExParams.device = context;
  encodeSessionExParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  encodeSessionExParams.apiVersion = NVENCAPI_VERSION;

  fan::cuda::check_error(func_list.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &hEncoder));

  NV_ENC_BUFFER_FORMAT fmt = NV_ENC_BUFFER_FORMAT_ARGB;

  NV_ENC_INITIALIZE_PARAMS initializeParams = { 0 };
  NV_ENC_CONFIG encConfig = { 0 };

  ZeroMemory(&initializeParams, sizeof(initializeParams));
  ZeroMemory(&initializeParams, sizeof(encConfig));
  encConfig.gopLength = 5;

  initializeParams.encodeConfig = &encConfig;


  CreateDefaultEncoderParams(&initializeParams, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_LOW_LATENCY_HP_GUID);
  //encInitParams.enableOutputInVidmem;

 /* initializeParams.encodeGUID = NV_ENC_CODEC_H264_GUID;
  initializeParams.encodeWidth = width;
  initializeParams.encodeHeight = height;
  initializeParams.maxEncodeWidth = width;
  initializeParams.maxEncodeHeight = height;
  initializeParams.darHeight = height;
  initializeParams.darWidth = width;
  initializeParams.frameRateNum = fps;
  initializeParams.frameRateDen = 1;
  initializeParams.enableEncodeAsync = 0;
  initializeParams.enablePTD = 1;
  initializeParams.reportSliceOffsets = 0;
  initializeParams.enableSubFrameWrite = 0;

  NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
  encodeConfig.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
  encodeConfig.gopLength = fps;
  encodeConfig.frameIntervalP = 1;
  encodeConfig.monoChromeEncoding = 0;
  encodeConfig.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
  encodeConfig.mvPrecision = NV_ENC_MV_PRECISION_DEFAULT;

  encodeConfig.rcParams.version = NV_ENC_RC_PARAMS_VER;
  encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
  encodeConfig.rcParams.constQP.qpInterB = 28;
  encodeConfig.rcParams.constQP.qpInterP = 26;
  encodeConfig.rcParams.constQP.qpIntra = 25;
  encodeConfig.rcParams.averageBitRate = bitrate;
  encodeConfig.rcParams.maxBitRate = bitrate * 2;

  initializeParams.encodeConfig = &encodeConfig;*/
  // Initialize encode configuration

  // Initialize preset configuration
  //presetConfig.version = NV_ENC_PRESET_CONFIG_VER;
  //presetConfig.presetCfg.version = NV_ENC_CONFIG_VER;
  //presetConfig.presetCfg.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
  //presetConfig.presetCfg.gopLength = fps;
  //presetConfig.presetCfg.frameIntervalP = 1;
  //presetConfig.presetCfg.monoChromeEncoding = 0;
  //presetConfig.presetCfg.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
  //presetConfig.presetCfg.mvPrecision = NV_ENC_MV_PRECISION_QUARTER_PEL;
  //presetConfig.presetCfg.rcParams.version = NV_ENC_RC_PARAMS_VER;
  //presetConfig.presetCfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
  //presetConfig.presetCfg.rcParams.constQP.qpInterB = 28;
  //presetConfig.presetCfg.rcParams.constQP.qpInterP = 26;
  //presetConfig.presetCfg.rcParams.constQP.qpIntra = 25;
  //presetConfig.presetCfg.rcParams.averageBitRate = bitrate;
  //presetConfig.presetCfg.rcParams.maxBitRate = bitrate * 2;

  //// Open the output file
  //output.open("output.h264", std::ios::out | std::ios::binary);
  //if (!output.is_open()) {
  //  std::cout << "Failed to open output file." << std::endl;
  //  exit(1);
  //}

  auto m_nEncoderBuffer = initializeParams.encodeConfig->frameIntervalP + initializeParams.encodeConfig->rcParams.lookaheadDepth;

  // Initialize the encoder
  fan::cuda::check_error(func_list.nvEncInitializeEncoder(hEncoder, &initializeParams));

  AllocateInputBuffers(m_nEncoderBuffer);

  std::vector<std::vector<uint8_t>> vPacket;

  m_vMappedInputBuffers.resize(m_nEncoderBuffer, nullptr);

  m_vBitstreamOutputBuffer.resize(m_nEncoderBuffer, nullptr);

  int i = m_iToSend % m_nEncoderBuffer;
  NV_ENC_MAP_INPUT_RESOURCE mapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };
  mapInputResource.registeredResource = m_vRegisteredResources[i];
  fan::cuda::check_error(func_list.nvEncMapInputResource(hEncoder, &mapInputResource));
  m_vMappedInputBuffers[i] = mapInputResource.mappedResource;
  DoEncode(m_vMappedInputBuffers[i], vPacket, nullptr);

  //DoEncode()

  //// Allocate input and output buffers
  //NV_ENC_INPUT_PTR inputBuffer = 0;
  //NV_ENC_OUTPUT_PTR outputBuffer = 0;

  //// Encode the RGBA frames
  //for (int i = 0; i < 100; i++) {
  //  // Fill the RGBA frame with random data
  //  memset(rgba, i % 256, frame_size);

  //  // Map the input buffer
  //  NV_ENC_MAP_INPUT_RESOURCE mapInputResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };
  //  mapInputResource.inputResource = rgba;
  //  //mapInputResource.mappedResource = &mappedResource;

  //  func_list.nvEncMapInputResource(&hEncoder, &mapInputResource);
  //  inputBuffer = mapInputResource.mappedResource;

  //  // Get an output buffer
  //  NV_ENC_LOCK_BITSTREAM lockBitstreamData = { 0 };
  //  lockBitstreamData.version = NV_ENC_LOCK_BITSTREAM_VER;
  //  lockBitstreamData.outputBitstream = outputBuffer;
  //  lockBitstreamData.doNotWait = false;

  //  // Encode the RGBA frame
  //  NV_ENC_PIC_PARAMS encodePicParams = { 0 };
  //  encodePicParams.version = NV_ENC_PIC_PARAMS_VER;
  //  encodePicParams.inputBuffer = inputBuffer;
  //  encodePicParams.bufferFmt = NV_ENC_BUFFER_FORMAT_ABGR;
  //  encodePicParams.inputWidth = width;
  //  encodePicParams.inputHeight = height;
  //  encodePicParams.outputBitstream = &lockBitstreamData;
  //  encodePicParams.completionEvent = 0;

  //  fan::cuda::check_error(func_list.nvEncEncodePicture(&hEncoder, &encodePicParams));

  //  // Unmap the input buffer
  //  //NV_ENC_INPUT_PTR unmapInputResource = { 0 };
  //  //unmapInputResource.version = NV_ENC_UNMAP_INPUT_RESOURCE_VER;
  //  //unmapInputResource.mappedResource = inputBuffer;

  //  //encoder.nvEncUnmapInputResource(&encoder, &unmapInputResource);

  //  // Write the encoded frame to the output file
  //  //output.write((char*)lockBitstreamData.bitstreamBuffer, lockBitstreamData.bitstreamSizeInBytes);

  //  // Release the output buffer
  //  fan::cuda::check_error(func_list.nvEncUnlockBitstream(&hEncoder, &lockBitstreamData));
  //}

  // Close the output file
  output.close();

  // Destroy the encoder
  //CHECK_NVENC_ERROR(encoder->nvEncDestroyEncoder(encoder));

  // Free the RGBA frame buffer
  delete[] rgba;

  return 0;
}