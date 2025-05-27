#pragma once

#ifndef __GPU_IS_CUDA_INITED
bool __GPU_IS_CUDA_INITED = 0;
#define __GPU_IS_CUDA_INITED __GPU_IS_CUDA_INITED
#endif

//extern void call_kernel(cudaSurfaceObject_t surface, int, int);

namespace fan {
  namespace cuda {

    struct nv_decoder_t {

      nv_decoder_t(loco_t* loco) {

        if (!__GPU_IS_CUDA_INITED) {
          fan::cuda::check_error(cuInit(0));
          __GPU_IS_CUDA_INITED = true;
        }

        int device_count = 0;
        fan::cuda::check_error(cuDeviceGetCount(&device_count));

        fan::print(device_count, "cuda device(s) found");

        fan::cuda::check_error(cuDeviceGet(&device, 0));

        char name[80] = { 0 };
        fan::cuda::check_error(cuDeviceGetName(name, sizeof(name), device));

        fan::print("cuda device: ", name);

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
      }

      ~nv_decoder_t() {
        image_y_resource.~graphics_resource_t();
        image_vu_resource.~graphics_resource_t();
        cuvidDestroyVideoParser(parser);

        fan::cuda::check_error(cuvidDestroyDecoder(decoder));
        fan::cuda::check_error(cuCtxDestroy(context));
      }

      void start_decoding(const std::string& nal_data) {
        CUVIDSOURCEDATAPACKET pkt;
        pkt.flags = 0;
        pkt.payload_size = nal_data.size();
        pkt.payload = (uint8_t*)nal_data.data();
        pkt.timestamp = 0;
        fan::cuda::check_error(cuvidParseVideoData(parser, &pkt));

        CUVIDSOURCEDATAPACKET eos_pkt = {};
        eos_pkt.flags = CUVID_PKT_ENDOFSTREAM;
        eos_pkt.payload_size = 0;
        eos_pkt.payload = nullptr;
        eos_pkt.timestamp = 0;
        fan::cuda::check_error(cuvidParseVideoData(parser, &eos_pkt));
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

        fan::print("coded size: ", std::to_string(fmt->coded_width) + "x" + std::to_string(fmt->coded_height));
        fan::print("display area: ", fmt->display_area.left, fmt->display_area.top, fmt->display_area.right, fmt->display_area.bottom);
        fan::print("bitrate ", fmt->bitrate);

        bool resized = decoder->frame_size != fan::vec2ui(fmt->coded_width, fmt->coded_height);

        decoder->frame_size = fan::vec2ui(fmt->coded_width, fmt->coded_height);

        int nDecodeSurface = GetNumDecodeSurfaces(fmt->codec, fmt->coded_width, fmt->coded_height);

        //cuvidDestroyVideoParser(decoder->parser);

        /*fan::cuda::check_error(cuvidDestroyDecoder(decoder->decoder));
        decoder->decoder = nullptr;*/

        if (decoder->decoder) {

          if (resized) {
            fan::cuda::check_error(cuvidDestroyDecoder(decoder->decoder));
            decoder->decoder = nullptr;
            goto g_remake_decoder;
          }

          CUVIDRECONFIGUREDECODERINFO reconfigParams = { 0 };

          reconfigParams.ulWidth = fmt->coded_width;
          reconfigParams.ulHeight = fmt->coded_height;
          reconfigParams.ulTargetWidth = reconfigParams.ulWidth;
          reconfigParams.ulTargetHeight = reconfigParams.ulHeight;

          reconfigParams.ulNumDecodeSurfaces = nDecodeSurface;

          //cuCtxPushCurrent(decoder->context);
          fan::cuda::check_error(cuvidReconfigureDecoder(decoder->decoder, &reconfigParams));

          //cuCtxPopCurrent(NULL);
        }
        else {
        g_remake_decoder:

          decoder->image_y_resource.close();
          decoder->image_vu_resource.close();

          fan::graphics::image_load_properties_t lp;
          // cudaGraphicsGLRegisterImage accepts only GL_RED
          lp.internal_format = fan::graphics::image_format::r8_unorm;
          lp.format = lp.internal_format;
          lp.min_filter = fan::graphics::image_filter::linear;
          lp.mag_filter = lp.min_filter;
          lp.visual_output = fan::graphics::image_sampler_address_mode::clamp_to_edge;
          fan::image::image_info_t image_info;
          image_info.data = 0;
          image_info.size = decoder->frame_size;
          decoder->images[0] = gloco->image_load(image_info, lp);
          decoder->image_y_resource.open(gloco->image_get_handle(decoder->images[0]));

          lp.internal_format = fan::graphics::image_format::rg8_unorm;
          lp.format = lp.internal_format;
          image_info.size /= 2;
          decoder->images[1] = gloco->image_load(image_info, lp);
          decoder->image_vu_resource.open(gloco->image_get_handle(decoder->images[1]));

          decoder->sequence_cb();

          CUVIDDECODECREATEINFO create_info = { 0 };
          create_info.CodecType = fmt->codec;
          create_info.ChromaFormat = fmt->chroma_format;
          create_info.OutputFormat = fmt->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
          create_info.bitDepthMinus8 = fmt->bit_depth_luma_minus8;
          create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
          create_info.ulNumOutputSurfaces = 1;

          create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
          create_info.ulNumDecodeSurfaces = nDecodeSurface;

          create_info.ulWidth = fmt->coded_width;
          create_info.ulHeight = fmt->coded_height;
          create_info.ulMaxWidth = create_info.ulWidth;
          create_info.ulMaxHeight = create_info.ulHeight;
          create_info.ulTargetWidth = create_info.ulWidth;
          create_info.ulTargetHeight = create_info.ulHeight;


          fan::cuda::check_error(cuvidCreateDecoder(&decoder->decoder, &create_info));
        }
        //

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
        //fan::print(info->picture_index);
        fan::cuda::check_error(cuvidMapVideoFrame(decoder->decoder, info->picture_index, &cuDevPtr,
          &nPitch, &videoProcessingParameters));

        fan::cuda::check_error(cudaMemcpy2DToArray(decoder->image_y_resource.cuda_array, 0, 0, (void*)cuDevPtr, nPitch, decoder->frame_size.x, decoder->frame_size.y, cudaMemcpyDeviceToDevice));
        fan::cuda::check_error(cudaMemcpy2DToArray(decoder->image_vu_resource.cuda_array, 0, 0, (void*)(cuDevPtr + nPitch * decoder->frame_size.y), nPitch, decoder->frame_size.x, decoder->frame_size.y / 2, cudaMemcpyDeviceToDevice));

        pile->loco.process_loop([]{});

        fan::delay(fan::time::nanoseconds(.033e+9));
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

      fan::graphics::image_t images[2];

      loco_t::cuda_textures_t::graphics_resource_t image_y_resource;
      loco_t::cuda_textures_t::graphics_resource_t image_vu_resource;

      std::function<void()> sequence_cb = [] {};
    };
  }
}