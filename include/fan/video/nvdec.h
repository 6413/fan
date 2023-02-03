#pragma once

#include <cuda.h>

#include <nvcuvid.h>

//#include "device_launch_parameters.h"

typedef void* HGPUNV;
//#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifndef __GPU_IS_CUDA_INITED
bool __GPU_IS_CUDA_INITED = 0;
#define __GPU_IS_CUDA_INITED __GPU_IS_CUDA_INITED
#endif

extern void call_kernel(cudaSurfaceObject_t surface, int, int);

namespace fan {
  namespace cuda {

    void check_error(auto result) {
      if (result != CUDA_SUCCESS) {
        if constexpr (std::is_same_v<decltype(result), CUresult>) {
          const char* err_str = nullptr;
          cuGetErrorString(result, &err_str);
          fan::throw_error("function failed with:" + std::to_string(result) + ", " + err_str);
        }
        else {
          fan::throw_error("function failed with:" + std::to_string(result) + ", ");
        }
      }
    }

    struct graphics_resource_t {
      graphics_resource_t() = default;

      void open(GLuint image_id) {
        fan::cuda::check_error(cudaGraphicsGLRegisterImage(&resource, image_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
      }

      void map(cudaArray_t* carray) {
        fan::cuda::check_error(cudaGraphicsMapResources(1, &resource, 0));
        fan::cuda::check_error(cudaGraphicsSubResourceGetMappedArray(carray, resource, 0, 0));
      }
      void unmap() {
        fan::cuda::check_error(cudaGraphicsUnmapResources(1, &resource));
      }
      ~graphics_resource_t() {
        if (resource == nullptr) {
          return;
        }
        cudaGraphicsUnregisterResource(resource);
      }
      cudaGraphicsResource* resource = nullptr;
    };

    struct nv_decoder_t {

      nv_decoder_t() {

        if (!__GPU_IS_CUDA_INITED) {
          fan::cuda::check_error(cuInit(0));
          __GPU_IS_CUDA_INITED = true;
        }

        int device_count = 0;
        fan::cuda::check_error(cuDeviceGetCount(&device_count));

        fan::print_format("{} cuda device(s) found", device_count);

        fan::cuda::check_error(cuDeviceGet(&device, 0));

        char name[80] = { 0 };
        fan::cuda::check_error(cuDeviceGetName(name, sizeof(name), device));

        fan::print_format("cuda device: {}", name);

        fan::cuda::check_error(cuCtxCreate(&context, 0, device));

        /* Query capabilities. */
        CUVIDDECODECAPS decode_caps = {};
        decode_caps.eCodecType = cudaVideoCodec_H264;
        decode_caps.eChromaFormat = cudaVideoChromaFormat_420;
        decode_caps.nBitDepthMinus8 = 0;

        fan::cuda::check_error(cuvidGetDecoderCaps(&decode_caps));

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

        fan::cuda::check_error(cuvidCreateVideoParser(&parser, &parser_params));

        loco_t::image_t::load_properties_t lp;
        // cudaGraphicsGLRegisterImage accepts only GL_RED
        lp.internal_format = GL_RED;
        lp.format = GL_RED;
        lp.filter = loco_t::image_t::filter::linear;
        lp.visual_output = loco_t::image_t::sampler_address_mode::clamp_to_edge;
        fan::webp::image_info_t info;
        info.data = 0;
        info.size = 0;
        image_y.load(&pile->loco, info, lp);
        // cudaGraphicsGLRegisterImage accepts only GL_RG
        lp.internal_format = fan::opengl::GL_RG;
        lp.format = fan::opengl::GL_RG;
        image_vu.load(&pile->loco, info, lp);
      }

      ~nv_decoder_t() {
        cuvidDestroyVideoParser(parser);

        fan::cuda::check_error(cuvidDestroyDecoder(decoder));
        fan::cuda::check_error(cuCtxDestroy(context));
      }

      void start_decoding(const fan::string& nal_data) {
        CUVIDSOURCEDATAPACKET pkt;
        pkt.flags = 0;
        pkt.payload_size = nal_data.size();
        pkt.payload = (uint8_t*)nal_data.data();
        pkt.timestamp = 0;
        fan::cuda::check_error(cuvidParseVideoData(parser, &pkt));
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

        fan::print_format("coded size: {}x{}", fmt->coded_width, fmt->coded_height);
        fan::print_format("display area: {}x{}", fmt->display_area.left, fmt->display_area.top, fmt->display_area.right, fmt->display_area.bottom);
        fan::print_format("birate {}", fmt->bitrate);

        CUVIDDECODECREATEINFO create_info = { 0 };
        create_info.CodecType = fmt->codec;
        create_info.ChromaFormat = fmt->chroma_format;
        create_info.OutputFormat = fmt->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
        create_info.bitDepthMinus8 = fmt->bit_depth_luma_minus8;
        create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
        create_info.ulNumOutputSurfaces = 1;

        create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
        int nDecodeSurface = GetNumDecodeSurfaces(fmt->codec, fmt->coded_width, fmt->coded_height);
        create_info.ulNumDecodeSurfaces = nDecodeSurface;

        create_info.ulWidth = fmt->coded_width;
        create_info.ulHeight = fmt->coded_height;
        create_info.ulMaxWidth = fmt->coded_width;
        create_info.ulMaxHeight = fmt->coded_height;
        create_info.ulTargetWidth = create_info.ulWidth;
        create_info.ulTargetHeight = create_info.ulHeight;

        decoder->frame_size = fan::vec2ui(fmt->coded_width, fmt->coded_height);

        loco_t::image_t::load_properties_t lp;
        // cudaGraphicsGLRegisterImage accepts only GL_RED
        lp.internal_format = GL_RED;
        lp.format = GL_RED;
        lp.filter = loco_t::image_t::filter::linear;
        lp.visual_output = loco_t::image_t::sampler_address_mode::clamp_to_edge;
        fan::webp::image_info_t info;
        info.data = 0;
        info.size = decoder->frame_size;
        decoder->image_y.reload_pixels(&pile->loco, info, lp);
        // cudaGraphicsGLRegisterImage accepts only GL_RG
        lp.internal_format = fan::opengl::GL_RG;
        lp.format = fan::opengl::GL_RG;
        decoder->image_vu.reload_pixels(&pile->loco, info, lp);

        decoder->image_y_resource.open(pile->loco.image_list[decoder->image_y.texture_reference].texture_id);
        decoder->image_vu_resource.open(pile->loco.image_list[decoder->image_vu.texture_reference].texture_id);

        fan::cuda::check_error(cuvidCreateDecoder(&decoder->decoder, &create_info));

        decoder->timestamp.start();

        return nDecodeSurface;
      }

      static int parser_decode_picture_callback(void* user, CUVIDPICPARAMS* pic) {
        nv_decoder_t* decoder = (nv_decoder_t*)user;

        if (nullptr == decoder->decoder) {
          fan::throw_error("decoder is null");
        }

        fan::cuda::check_error(cuvidDecodePicture(decoder->decoder, pic));

        return 1;
      }

      static int parser_display_picture_callback(void* user, CUVIDPARSERDISPINFO* info) {
        nv_decoder_t* decoder = (nv_decoder_t*)user;

        unsigned long long cuDevPtr = 0;
        unsigned int nPitch;

        CUVIDPROCPARAMS videoProcessingParameters = {};
        videoProcessingParameters.progressive_frame = info->progressive_frame;
        videoProcessingParameters.second_field = info->repeat_first_field + 1;
        videoProcessingParameters.top_field_first = info->top_field_first;
        videoProcessingParameters.unpaired_field = info->repeat_first_field < 0;

        fan::cuda::check_error(cuvidMapVideoFrame(decoder->decoder, info->picture_index, &cuDevPtr,
          &nPitch, &videoProcessingParameters));


        cudaArray_t mappedArray;

        decoder->image_y_resource.map(&mappedArray);
        fan::cuda::check_error(cudaMemcpy2DToArray(mappedArray, 0, 0, (void*)cuDevPtr, nPitch, decoder->frame_size.x, decoder->frame_size.y, cudaMemcpyDeviceToDevice));

        decoder->image_vu_resource.map(&mappedArray);
        fan::cuda::check_error(cudaMemcpy2DToArray(mappedArray, 0, 0, (void*)(cuDevPtr + nPitch * decoder->frame_size.y), nPitch, decoder->frame_size.x, decoder->frame_size.y / 2, cudaMemcpyDeviceToDevice));

        decoder->image_y_resource.unmap();
        decoder->image_vu_resource.unmap();

        pile->loco.process_loop([]{});

        ////fan::delay(fan::time::nanoseconds(.05e+9));
        fan::cuda::check_error(cuvidUnmapVideoFrame(decoder->decoder, cuDevPtr));

        decoder->current_frame++;

        return 1;
      }

      CUcontext context = { 0 };
      CUvideodecoder decoder = nullptr;
      CUdevice device = { 0 };
      CUvideoparser parser = nullptr;
      fan::vec2ui frame_size;

      fan::time::clock timestamp;
      uint32_t current_frame = 0;

      fan::cuda::graphics_resource_t image_y_resource;
      fan::cuda::graphics_resource_t image_vu_resource;

      loco_t::image_t image_y;
      loco_t::image_t image_vu;
    };
  }
}