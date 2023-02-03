// Creates window, opengl context and renders a rectangle

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
#include <cuda.h>

#include <nvcuvid.h>

#include "device_launch_parameters.h"

#define HGPUNV void*
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


extern void call_kernel(cudaSurfaceObject_t surface, int, int);

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

pile_t* pile = new pile_t;
loco_t::nv12_t::properties_t p;
loco_t::nv12_t::properties_t sp;

fan::time::clock c;
int count = 0;

cudaGraphicsResource* image_y_resource;
cudaGraphicsResource* image_vu_resource;

loco_t::image_t image_y;
loco_t::image_t image_vu;

cudaArray_t d_array;

struct nv_decoder_t {

  CUcontext context = { 0 };
  CUvideodecoder decoder = nullptr;
  CUdevice device = { 0 };
  CUvideoctxlock lock;
  CUvideoparser parser = nullptr;
  fan::vec2ui frame_size;
  GLuint pbo;

  nv_decoder_t() {
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

    /* Create a video parser that gives us the CUVIDPICPARAMS structures. */
    CUVIDPARSERPARAMS parser_params;
    memset((void*)&parser_params, 0x00, sizeof(parser_params));
    parser_params.CodecType = cudaVideoCodec_H264;
    parser_params.ulMaxNumDecodeSurfaces = 1;
    parser_params.ulErrorThreshold = 0;
    parser_params.ulMaxDisplayDelay = 0;
    parser_params.pUserData = this;
    parser_params.pfnSequenceCallback = parser_sequence_callback;
    parser_params.pfnDecodePicture = parser_decode_picture_callback;
    parser_params.pfnDisplayPicture = parser_display_picture_callback;

    r = cuvidCreateVideoParser(&parser, &parser_params);

    fan::string video_data;
    fan::io::file::read("o3.264", &video_data);

    CUVIDSOURCEDATAPACKET pkt;
    pkt.flags = 0;
    pkt.payload_size = video_data.size();
    pkt.payload = (uint8_t*)video_data.data();
    pkt.timestamp = 0;

    check_error(cuvidParseVideoData(parser, &pkt));
  }

  ~nv_decoder_t() {
    if (h_frame) {
      delete[] h_frame;
    }
  }

  static unsigned long GetNumDecodeSurfaces(cudaVideoCodec eCodec, unsigned int nWidth, unsigned int nHeight) {
    if (eCodec == cudaVideoCodec_VP9) {
      return 12;
    }

    if (eCodec == cudaVideoCodec_H264 || eCodec == cudaVideoCodec_H264_SVC || eCodec == cudaVideoCodec_H264_MVC) {
      // assume worst-case of 20 decode surfaces for H264
      return 20;
    }

    if (eCodec == cudaVideoCodec_HEVC) {
      // ref HEVC spec: A.4.1 General tier and level limits
      // currently assuming level 6.2, 8Kx4K
      int MaxLumaPS = 35651584;
      int MaxDpbPicBuf = 6;
      int PicSizeInSamplesY = (int)(nWidth * nHeight);
      int MaxDpbSize;
      if (PicSizeInSamplesY <= (MaxLumaPS >> 2))
        MaxDpbSize = MaxDpbPicBuf * 4;
      else if (PicSizeInSamplesY <= (MaxLumaPS >> 1))
        MaxDpbSize = MaxDpbPicBuf * 2;
      else if (PicSizeInSamplesY <= ((3 * MaxLumaPS) >> 2))
        MaxDpbSize = (MaxDpbPicBuf * 4) / 3;
      else
        MaxDpbSize = MaxDpbPicBuf;
      return (std::min)(MaxDpbSize, 16) + 4;
    }

    return 8;
  }


  static int parser_sequence_callback(void* user, CUVIDEOFORMAT* fmt) {

    nv_decoder_t* decoder = (nv_decoder_t*)user;

    printf("CUVIDEOFORMAT.Coded size: %d x %d\n", fmt->coded_width, fmt->coded_height);
    printf("CUVIDEOFORMAT.Display area: %d %d %d %d\n", fmt->display_area.left, fmt->display_area.top, fmt->display_area.right, fmt->display_area.bottom);
    printf("CUVIDEOFORMAT.Bitrate: %u\n", fmt->bitrate);

    check_error(cuvidCtxLockCreate(&decoder->lock, decoder->context));

    CUVIDDECODECREATEINFO create_info = { 0 };
    create_info.CodecType = fmt->codec;
    create_info.ChromaFormat = fmt->chroma_format;
    create_info.OutputFormat = fmt->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
    create_info.bitDepthMinus8 = fmt->bit_depth_luma_minus8;
    create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    create_info.ulNumOutputSurfaces = 1;
    // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by NVDEC hardware
    create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    int nDecodeSurface = GetNumDecodeSurfaces(fmt->codec, fmt->coded_width, fmt->coded_height);
    create_info.ulNumDecodeSurfaces = nDecodeSurface;
    //create_info.vidLock = decoder->lock;
    create_info.ulWidth = fmt->coded_width;
    create_info.ulHeight = fmt->coded_height;
    create_info.ulMaxWidth = fmt->coded_width;
    create_info.ulMaxHeight = fmt->coded_height;
    create_info.ulTargetWidth = create_info.ulWidth;                   /* Post-processed output width (should be aligned to 2). */
    create_info.ulTargetHeight = create_info.ulHeight;

    decoder->frame_size = fan::vec2ui(fmt->coded_width, fmt->coded_height);

    /* @todo do we need this? */
    /* create_info.vidLock = ...*/

    image_y.create_texture(&pile->loco);
    image_y.bind_texture(&pile->loco);
    loco_t::image_t::load_properties_t lp;
    lp.internal_format = GL_R;
    lp.format = GL_R;
    lp.filter = loco_t::image_t::filter::linear;
    fan::webp::image_info_t info;
    info.data = 0;
    info.size = decoder->frame_size;
    image_y.load(&pile->loco, info);
    lp.internal_format = fan::opengl::GL_RG;
    lp.format = fan::opengl::GL_RG;
    image_vu.load(&pile->loco, info);

    check_error(cudaGraphicsGLRegisterImage(&image_y_resource, pile->loco.image_list[image_y.texture_reference].texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
    check_error(cudaGraphicsGLRegisterImage(&image_vu_resource, pile->loco.image_list[image_vu.texture_reference].texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));

    check_error(cuvidCreateDecoder(&decoder->decoder, &create_info));

    c.start();


    return nDecodeSurface;
  }

  static int parser_decode_picture_callback(void* user, CUVIDPICPARAMS* pic) {

    nv_decoder_t* decoder = (nv_decoder_t*)user;

    CUresult r = CUDA_SUCCESS;

    if (nullptr == decoder->decoder) {
      printf("decoder is nullptr. (exiting).");
      exit(EXIT_FAILURE);
    }

    r = cuvidDecodePicture(decoder->decoder, pic);
    if (CUDA_SUCCESS != r) {
      printf("Failed to decode the picture.");
    }

    return 1;
  }

  //fan::vec2 ress[] = {}

  bool init = false;

  unsigned char* h_frame = 0;

  static int parser_display_picture_callback(void* user, CUVIDPARSERDISPINFO* info) {


    nv_decoder_t* decoder = (nv_decoder_t*)user;

    unsigned long long cuDevPtr = 0;
    unsigned int nPitch, frameSize;
    unsigned char* pHostPtr = nullptr;

    CUVIDPROCPARAMS videoProcessingParameters = {};
    videoProcessingParameters.progressive_frame = info->progressive_frame;
    videoProcessingParameters.second_field = info->repeat_first_field + 1;
    videoProcessingParameters.top_field_first = info->top_field_first;
    videoProcessingParameters.unpaired_field = info->repeat_first_field < 0;
    //videoProcessingParameters.output_stream = m_cuvidStream;


    check_error(cuvidCtxLock(decoder->lock, 0));

    check_error(cuvidMapVideoFrame(decoder->decoder, info->picture_index, &cuDevPtr,
      &nPitch, &videoProcessingParameters));


    cudaArray_t mappedArray;
    cudaArray_t mappedArray2;
    check_error(cudaGraphicsMapResources(1, &image_y_resource, 0));
    check_error(cudaGraphicsSubResourceGetMappedArray(&mappedArray, image_y_resource, 0, 0));

    check_error(cudaGraphicsMapResources(1, &image_vu_resource, 0));
    check_error(cudaGraphicsSubResourceGetMappedArray(&mappedArray2, image_vu_resource, 0, 0));


    check_error(cudaMemcpy2DToArray(mappedArray, 0, 0, (void*)cuDevPtr, nPitch, decoder->frame_size.x, decoder->frame_size.y, cudaMemcpyDeviceToDevice));
    check_error(cudaMemcpy2DToArray(mappedArray2, 0, 0, (void*)(cuDevPtr + nPitch * decoder->frame_size.y), nPitch, decoder->frame_size.x, decoder->frame_size.y / 2, cudaMemcpyDeviceToDevice));

    check_error(cudaGraphicsUnmapResources(1, &image_y_resource));
    check_error(cudaGraphicsUnmapResources(1, &image_vu_resource));

    //fan::io::file::write("help", fan::string((uint8_t*)h_frame, (uint8_t*)h_frame + frameSize), std::ios_base::binary);

    if (!decoder->init) {


      uint64_t offset = 0;
      void* data[2];
      data[0] = decoder->h_frame;
      data[1] = (uint8_t*)decoder->h_frame + (uint64_t)decoder->frame_size.multiply();
      p.y = &image_y;
      p.vu = &image_vu;
      p.tc_size = fan::vec2(0.25, 1);
      //p.image = &image;
      pile->loco.nv12.push_back(&pile->cid[1], p);
      decoder->init = true;
    }

    pile->loco.process_loop([&] {});

    fan::delay(fan::time::nanoseconds(.016e+9));
    check_error(cuvidUnmapVideoFrame(decoder->decoder, cuDevPtr));


    check_error(cuvidCtxUnlock(decoder->lock, 0));
    count++;
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
};

int main() {

  p.size = fan::vec2(1);
  p.matrices = &pile->matrices;
  p.viewport = &pile->viewport;

  //sp.size = 1;
  //sp.matrices = &pile->matrices;
  //sp.viewport = &pile->viewport;
  //sp.position = 0;

  pile->loco.set_vsync(false);
  //pile->loco.get_window()->add_keys_callback([](const auto& d) {
  //  if (d.key == fan::key_up) {
  //    fan::print("a");
  //  }
  //  });
  nv_decoder_t nv;
  fan::print(c.elapsed(), count, c.elapsed() / count);



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