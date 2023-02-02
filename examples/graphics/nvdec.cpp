// Creates window, opengl context and renders a rectangle

#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 3
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

//#define loco_vulkan


#define loco_window
#define loco_context

//#define loco_rectangle
#define loco_nv12
#include _FAN_PATH(graphics/loco.h)

#include <cuda.h>
#include <nvcuvid.h>

#define HGPUNV void*
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

fan::vec2 res = fan::vec2(576, 360);

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
  fan::graphics::cid_t cid[(unsigned long long)1e+7];
};

void check_error(auto result) {
  if (result != CUDA_SUCCESS) {
    fan::throw_error("function failed with:" + std::to_string(result));
  }
}

CUcontext context = { 0 };
CUvideodecoder decoder = nullptr;
CUdevice device = { 0 };

CUdeviceptr d_cudaRGBA;
cudaGraphicsResource_t cudaResource;

pile_t* pile = new pile_t;
loco_t::nv12_t::properties_t p;

static int parser_sequence_callback(void* user, CUVIDEOFORMAT* fmt) {
  printf("CUVIDEOFORMAT.Coded size: %d x %d\n", fmt->coded_width, fmt->coded_height);
  printf("CUVIDEOFORMAT.Display area: %d %d %d %d\n", fmt->display_area.left, fmt->display_area.top, fmt->display_area.right, fmt->display_area.bottom);
  printf("CUVIDEOFORMAT.Bitrate: %u\n", fmt->bitrate);


  //check_error(cudaMalloc((void**)&d_cudaRGBA, fmt->coded_width * fmt->coded_height * 4));

  //GLuint pbo;
  //pile->loco.get_context()->opengl.glGenBuffers(1, &pbo);
  //pile->loco.get_context()->opengl.glBindBuffer(fan::opengl::GL_PIXEL_UNPACK_BUFFER, pbo);
  //pile->loco.get_context()->opengl.glBufferData(fan::opengl::GL_PIXEL_UNPACK_BUFFER, fmt->coded_width * fmt->coded_height * 4, NULL, fan::opengl::GL_STREAM_DRAW);

  //// Register CUDA buffer with OpenGL
  //check_error(cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, CU_GRAPHICS_REGISTER_FLAGS_NONE));

  return 0;
}

static int parser_decode_picture_callback(void* user, CUVIDPICPARAMS* pic) {

  CUresult r = CUDA_SUCCESS;

  if (nullptr == decoder) {
    printf("decoder is nullptr. (exiting).");
    exit(EXIT_FAILURE);
  }

  r = cuvidDecodePicture(decoder, pic);
  if (CUDA_SUCCESS != r) {
    printf("Failed to decode the picture.");
  }

  return 1;
}

//fan::vec2 ress[] = {}

bool init = false;

static int parser_display_picture_callback(void* user, CUVIDPARSERDISPINFO* info) {

  //CUVIDPARSEDISPINFO stDispInfo;
  CUVIDPROCPARAMS stProcParams;
  CUresult rResult;
  unsigned long long cuDevPtr = 0;
  unsigned int nPitch, frameSize;
  unsigned char* pHostPtr = nullptr;
  memset(&stProcParams, 0, sizeof(CUVIDPROCPARAMS));
  


  rResult = cuvidMapVideoFrame(decoder, info->picture_index, &cuDevPtr,
    &nPitch, &stProcParams);

  CUVIDGETDECODESTATUS DecodeStatus;
  memset(&DecodeStatus, 0, sizeof(DecodeStatus));
  CUresult result = cuvidGetDecodeStatus(decoder, info->picture_index, &DecodeStatus);
  if (result == CUDA_SUCCESS && (DecodeStatus.decodeStatus == cuvidDecodeStatus_Error || DecodeStatus.decodeStatus == cuvidDecodeStatus_Error_Concealed))
  {
    printf("Decode Error occurred for picture %d\n", info->picture_index);
  }

  printf("+ mapping: %u succeeded\n", info->picture_index);

  frameSize = res.x * res.y + res.x * res.y / 2;
  unsigned char* h_frame = new unsigned char[frameSize];
  cudaMemcpy2D(h_frame, res.x, (void*)cuDevPtr, nPitch, res.x, res.y + res.y / 2, cudaMemcpyDeviceToHost);

  uint64_t offset = 0;
  void* data[2];
  data[0] = h_frame;
  data[1] = (uint8_t*)h_frame + (offset += res.multiply());

  if (!init) {

    p.load(&pile->loco, data, res);

    p.position = fan::vec2(0, 0);
    //p
    pile->loco.nv12.push_back(&pile->cid[0], p);
    init = true;
  }
  else {
    pile->loco.nv12.reload(&pile->cid[0], data, res);
  }

  pile->loco.process_loop([] {});

  //static int x = 1;
  //fan::print(x++);
  //if (x >= 200) {
  //fan::io::file::write("help", fan::string((uint8_t*)h_frame, (uint8_t*)h_frame + frameSize), std::ios_base::binary);
  //  fan::print("a");
  //}
  delete[] h_frame;
  cuvidUnmapVideoFrame(decoder, cuDevPtr);
  fan::delay(fan::time::nanoseconds(.1e+9));
  return 1;
}

/* ------------------------------------------------ */

static void print_cuvid_decode_caps(CUVIDDECODECAPS* caps) {

  if (nullptr == caps) {
    printf("Cannot print the cuvid decode caps as the given pointer is a nullptr.");
    return;
  }

  printf("CUVIDDECODECAPS.nBitDepthMinus8: %u\n", caps->nBitDepthMinus8);
  printf("CUVIDDECODECAPS.bIsSupported: %u\n", caps->bIsSupported);
  printf("CUVIDDECODECAPS.nMaxWidth: %u\n", caps->nMaxWidth);
  printf("CUVIDDECODECAPS.nMaxHeight: %u\n", caps->nMaxHeight);
  printf("CUVIDDECODECAPS.nMaxMBCount: %u\n", caps->nMaxMBCount);
  printf("CUVIDDECODECAPS.nMinWidth: %u\n", caps->nMinWidth);
  printf("CUVIDDECODECAPS.nMinHeight: %u\n", caps->nMinHeight);
}

static void print_cuvid_parser_disp_info(CUVIDPARSERDISPINFO* info) {

  if (nullptr == info) {
    printf("Cannot print the cuvid parser disp info, nullptr given.");
    return;
  }

  printf("CUVIDPARSERDISPINFO.picture_index: %d\n", info->picture_index);
  printf("CUVIDPARSERDISPINFO.progressive_frame: %d\n", info->progressive_frame);
  printf("CUVIDPARSERDISPINFO.top_field_first: %d\n", info->top_field_first);
  printf("CUVIDPARSERDISPINFO.repeat_first_field: %d\n", info->repeat_first_field);
  printf("CUVIDPARSERDISPINFO.timestamp: %lld\n", info->timestamp);
}

static void print_cuvid_pic_params(CUVIDPICPARAMS* pic) {

  if (nullptr == pic) {
    printf("Cannot print the cuvid pic params, nullptr given.");
    return;
  }

  printf("CUVIDPICPARAMS.PicWithInMbs: %d\n", pic->PicWidthInMbs);
  printf("CUVIDPICPARAMS.FrameHeightInMbs: %d\n", pic->FrameHeightInMbs);
  printf("CUVIDPICPARAMS.CurrPicIdx: %d\n", pic->CurrPicIdx);
  printf("CUVIDPICPARAMS.field_pic_flag: %d\n", pic->field_pic_flag);
  printf("CUVIDPICPARAMS.bottom_field_flag: %d\n", pic->bottom_field_flag);
  printf("CUVIDPICPARAMS.second_field: %d\n", pic->second_field);
  printf("CUVIDPICPARAMS.nBitstreamDataLen: %u\n", pic->nBitstreamDataLen);
  printf("CUVIDPICPARAMS.nNumSlices: %u\n", pic->nNumSlices);
  printf("CUVIDPICPARAMS.ref_pic_flag: %d\n", pic->ref_pic_flag);
  printf("CUVIDPICPARAMS.intra_pic_flag: %d\n", pic->intra_pic_flag);
}


int main() {

  p.size = fan::vec2(1);
  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;

  pile->loco.set_vsync(false);

  CUresult r = CUDA_SUCCESS;
  const char* err_str = nullptr;

  /* Initialize cuda, must be done before anything else. */
  r = cuInit(0);
  if (CUDA_SUCCESS != r) {
    cuGetErrorString(r, &err_str);
    printf("Failed to initialize cuda: %s. (exiting).\n", err_str);
    exit(EXIT_FAILURE);
  }

  int device_count = 0;
  r = cuDeviceGetCount(&device_count);
  if (CUDA_SUCCESS != r) {
    cuGetErrorString(r, &err_str);
    printf("Failed to get the cuda device count: %s. (exiting).\n", err_str);
    exit(EXIT_FAILURE);
  }

  printf("We have %d cuda device(s).\n", device_count);

  r = cuDeviceGet(&device, 0);
  if (CUDA_SUCCESS != r) {
    cuGetErrorString(r, &err_str);
    printf("Failed to get a handle to the cuda device: %s. (exiting).\n", err_str);
    exit(EXIT_FAILURE);
  }

  char name[80] = { 0 };
  r = cuDeviceGetName(name, sizeof(name), device);
  if (CUDA_SUCCESS != r) {
    cuGetErrorString(r, &err_str);
    printf("Failed to get the cuda device name: %s. (exiting).\n", err_str);
    exit(EXIT_FAILURE);
  }

  printf("Cuda device: %s.\n", name);

  r = cuCtxCreate(&context, 0, device);
  if (CUDA_SUCCESS != r) {
    cuGetErrorString(r, &err_str);
    printf("Failed to create a cuda context: %s. (exiting).\n", err_str);
    exit(EXIT_FAILURE);
  }

  /* Query capabilities. */
  CUVIDDECODECAPS decode_caps = {};
  decode_caps.eCodecType = cudaVideoCodec_H264;
  decode_caps.eChromaFormat = cudaVideoChromaFormat_420;
  decode_caps.nBitDepthMinus8 = 0;

  r = cuvidGetDecoderCaps(&decode_caps);
  if (CUDA_SUCCESS != r) {
    cuGetErrorString(r, &err_str);
    printf("Failed to get decoder caps: %s (exiting).\n", err_str);
    exit(EXIT_FAILURE);
  }

  /* Create decoder context. */
  CUVIDDECODECREATEINFO create_info = { 0 };
  create_info.CodecType = decode_caps.eCodecType;                    /* cudaVideoCodex_XXX */
  create_info.ChromaFormat = decode_caps.eChromaFormat;              /* cudaVideoChromaFormat_XXX */
  create_info.OutputFormat = cudaVideoSurfaceFormat_NV12;            /* cudaVideoSurfaceFormat_XXX */
  create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;         /* cudaVideoCreate_XXX */
  create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;      /* cudaVideoDeinterlaceMode_XXX */
  create_info.bitDepthMinus8 = decode_caps.nBitDepthMinus8;;
  create_info.ulNumOutputSurfaces = 1;                               /* Maximum number of internal decode surfaces. */
  create_info.ulNumDecodeSurfaces = 1;                               /* @todo from NvDecoder.cpp, assuming worst case here ... Maximum number of internal decode surfaces. */
  create_info.ulIntraDecodeOnly = 0;                                 /* @todo this seems like an interesting flag. */

  /* Size is specific for the moonlight.264 file. */
  create_info.ulWidth = res.x;                                        /* Coded sequence width in pixels. */
  create_info.ulHeight = res.y;                                       /* Coded sequence height in pixels. */
  create_info.ulTargetWidth = create_info.ulWidth;                   /* Post-processed output width (should be aligned to 2). */
  create_info.ulTargetHeight = create_info.ulHeight;                 /* Post-processed output height (should be aligned to 2). */


  /* @todo do we need this? */
  /* create_info.vidLock = ...*/

  r = cuvidCreateDecoder(&decoder, &create_info);
  if (CUDA_SUCCESS != r) {
    cuGetErrorString(r, &err_str);
    printf("Failed to create the decoder: %s. (exiting).\n", err_str);
    exit(EXIT_FAILURE);
  }

  /* Create a video parser that gives us the CUVIDPICPARAMS structures. */
  CUVIDPARSERPARAMS parser_params;
  memset((void*)&parser_params, 0x00, sizeof(parser_params));
  parser_params.CodecType = create_info.CodecType;
  parser_params.ulMaxNumDecodeSurfaces = create_info.ulNumDecodeSurfaces;
  parser_params.ulClockRate = 0;
  parser_params.ulErrorThreshold = 0;
  parser_params.ulMaxDisplayDelay = 1;
  parser_params.pUserData = nullptr;
  parser_params.pfnSequenceCallback = parser_sequence_callback;
  parser_params.pfnDecodePicture = parser_decode_picture_callback;
  parser_params.pfnDisplayPicture = parser_display_picture_callback;

  CUvideoparser parser = nullptr;
  r = cuvidCreateVideoParser(&parser, &parser_params);

  if (CUDA_SUCCESS != r) {
    cuGetErrorString(r, &err_str);
    printf("Failed to create a video parser: %s (exiting).\n", err_str);
    exit(EXIT_FAILURE);
  }


  fan::string video_data;
  fan::io::file::read("o.264", &video_data);

  CUVIDSOURCEDATAPACKET pkt;
  pkt.flags = 0;
  pkt.payload_size = video_data.size();
  pkt.payload = (uint8_t*)video_data.data();
  pkt.timestamp = 0;

  r = cuvidParseVideoData(parser, &pkt);
  if (CUDA_SUCCESS != r) {
    cuGetErrorString(r, &err_str);
    printf("Failed to parse h264 packet: %s (exiting).\n", err_str);
    exit(EXIT_FAILURE);

  }



  //r = cuCtxDestroy(context);
  //if (CUDA_SUCCESS != r) {
  //  cuGetErrorString(r, &err_str);
  //  printf("Failed to cleanly destroy the cuda context: %s (exiting).\n", err_str);
  //  exit(EXIT_FAILURE);
  //}

  //r = cuvidDestroyDecoder(decoder);
  //if (CUDA_SUCCESS != r) {
  //  cuGetErrorString(r, &err_str);
  //  printf("Failed to cleanly destroy the decoder context: %s. (exiting).\n", err_str);
  //  exit(EXIT_FAILURE);
  //}

  //if (nullptr != parser) {
  //  r = cuvidDestroyVideoParser(parser);
  //  if (CUDA_SUCCESS != r) {
  //    cuGetErrorString(r, &err_str);
  //    printf("Failed to the video parser context: %s. (exiting).\n", err_str);
  //    exit(EXIT_FAILURE);
  //  }
  //}

  //context = nullptr;
  //decoder = nullptr;
  //parser = nullptr;

  return 0;
}