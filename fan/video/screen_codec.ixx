module;

#include <fan/types/types.h>
#include <WITCH/WITCH.h>
//#define ETC_VEDC_Encode_DefineEncoder_OpenH264
//#define ETC_VEDC_Encode_DefineEncoder_x264
//#if defined(__platform_windows)
#define ETC_VEDC_Encode_DefineEncoder_nvenc
//#endif

//#define ETC_VEDC_Decoder_DefineCodec_OpenH264
//#if defined(__platform_windows)
#define ETC_VEDC_Decoder_DefineCodec_cuvid
//#endif#

#include <WITCH/PR/PR.h>
#include <WITCH/STR/psh.h>
#include <WITCH/STR/psf.h>
#include <WITCH/MD/SCR/SCR.h>
#include <WITCH/MD/Mice.h>
#include <WITCH/MD/Keyboard/Keyboard.h>
#include <WITCH/T/T.h>

#include <WITCH/MEM/MEM.h>
#include <WITCH/STR/common/common.h>
#include <WITCH/IO/IO.h>
#include <WITCH/IO/print.h>
#include <WITCH/HASH/SHA.h>
#include <WITCH/RAND/RAND.h>
#include <WITCH/VEC/VEC.h>
#include <WITCH/EV/EV.h>
#include <WITCH/ETC/VEDC/encode.h>
#include <WITCH/ETC/VEDC/Decoder.h>

export module fan.graphics.video.screen_codec;

export import fan.graphics.loco;

export namespace fan {
  namespace graphics {
    struct screen_codec_t {
      screen_codec_t() {

        if (MD_SCR_open(&mdscr) != 0) {
          __abort();
        }

        encoder.settings = {
         .CodecStandard = ETC_VCODECSTD_H264,
         .UsageType = ETC_VEDC_EncoderSetting_UsageType_Realtime,
         .RateControl{
           .Type = ETC_VEDC_EncoderSetting_RateControlType_VBR,
           .VBR = {.bps = 16000000 }
         },
         .InputFrameRate = 30
        };

        encoder.settings.FrameSizeX = mdscr.Geometry.Resolution.x;
        encoder.settings.FrameSizeY = mdscr.Geometry.Resolution.y;

        ETC_VEDC_Encode_Error err = ETC_VEDC_Encode_Open(
          &encoder,
          encoder.name.size(),
          encoder.name.c_str(),
          &encoder.settings,
          NULL);
        if (err != ETC_VEDC_Encode_Error_Success) {
          fan::print("failed to open encoder ", err);
          /* TODO */
          __abort();
        }

        auto r = ETC_VEDC_Decoder_Open(
          &decoder,
          decoder.name.size(),
          decoder.name.c_str(),
          0);
        if (r != ETC_VEDC_Decoder_Error_OK) {


          fan::printn8(
            "[CLIENT] [WARNING] [DECODER] ", __FUNCTION__, " ", __FILE__, ":", __LINE__,
            " (ETC_VEDC_Decoder_Open returned (", r, ") for encoder \"",
            decoder.name, "\""
          );

          fan::printn8(
            "[CLIENT] [WARNING] [DECODER] ", __FUNCTION__, " ", __FILE__, ":", __LINE__,
            " falling back to OpenH264 decoder."
          );
        }

        new (&ReadMethodData.CudaArrayFrame) ReadMethodData_t::CudaArrayFrame_t;
      }

      struct : ETC_VEDC_Encode_t {
        std::string name = "nvenc";
        ETC_VEDC_EncoderSetting_t settings;
        void* data{};
        uintptr_t amount = 0;
        bool updated = true;
      }encoder;
      // returns if is readable
      bool encode_write() {
        if (
          encoder.settings.FrameSizeX != mdscr.Geometry.Resolution.x ||
          encoder.settings.FrameSizeY != mdscr.Geometry.Resolution.y
          ) {
          encoder.settings.FrameSizeX = mdscr.Geometry.Resolution.x;
          encoder.settings.FrameSizeY = mdscr.Geometry.Resolution.y;
          sint32_t err = ETC_VEDC_Encode_SetFrameSize(
            &encoder,
            encoder.settings.FrameSizeX,
            encoder.settings.FrameSizeY);
          if (err != 0) {
            /* TODO */
            __abort();
          }
        }

        {
          ETC_VEDC_Encode_Frame_t Frame;
          /* TOOD hardcode to spectific pixel format */
          Frame.Properties.PixelFormat = PIXF_BGRA;
          Frame.Properties.Stride[0] = mdscr.Geometry.LineSize;
          Frame.Properties.SizeX = mdscr.Geometry.Resolution.x;
          Frame.Properties.SizeY = mdscr.Geometry.Resolution.y;
          Frame.Data[0] = screen_buffer;
          Frame.TimeStamp = FrameProcessStartTime - EncoderStartTime;


          ETC_VEDC_Encode_Error err = ETC_VEDC_Encode_Write(&encoder, ETC_VEDC_Encode_WriteType_Frame, &Frame);
          if (err != ETC_VEDC_Encode_Error_Success) {
            /* TODO */
            __abort();
          }
        }

        return ETC_VEDC_Encode_IsReadable(&encoder) != false;
      }
      uintptr_t encode_read() {
        ETC_VEDC_Encode_PacketInfo PacketInfo;
        encoder.amount = ETC_VEDC_Encode_Read(&encoder, &PacketInfo, &encoder.data);
        return encoder.amount;
      }
      static auto get_encoders() {
        return std::to_array(_ETC_VEDC_EncoderList);
      }


      struct : ETC_VEDC_Decoder_t {
        std::string name = "cuvid";
        bool updated = true;
      }decoder;
      
      uintptr_t decode(loco_t::shape_t& universal_image_renderer) {
        sintptr_t r = ETC_VEDC_Decoder_Write(&decoder, (uint8_t*)encoder.data, encoder.amount);

        if (!ETC_VEDC_Decoder_IsReadable(&decoder)) {
          return 0;
        }
        if (!ETC_VEDC_Decoder_IsReadType(&decoder, ETC_VEDC_Decoder_ReadType_CudaArrayFrame)) {
          fan::throw_error("A");
        }

        NewReadMethod(universal_image_renderer, ETC_VEDC_Decoder_ReadType_CudaArrayFrame);

        CUcontext CudaContext = (CUcontext)ETC_VEDC_Decoder_GetUnique(
          &decoder,
          ETC_VEDC_Decoder_UniqueType_CudaContext);
        if (cuCtxSetCurrent(CudaContext) != CUDA_SUCCESS) {
          __abort();
        }

        ETC_VEDC_Decoder_ImageProperties_t ImageProperties;
        ETC_VEDC_Decoder_GetReadImageProperties(
          &decoder,
          ETC_VEDC_Decoder_ReadType_CudaArrayFrame,
          &ImageProperties);

        /* TODO hardcoded pixel format because fan uses different pixel format lib */
        ReadMethodData.CudaArrayFrame.cuda_textures.resize(
          gloco,
          universal_image_renderer,
          fan::graphics::image_format::nv12,
          fan::vec2ui(ImageProperties.SizeX, ImageProperties.SizeY));

        ETC_VEDC_Decoder_CudaArrayFrame_t Frame;
        for (uint32_t i = 0; i < 4; i++) {
          Frame.Array[i] = ReadMethodData.CudaArrayFrame.cuda_textures.get_array(i);
        }

        if (ETC_VEDC_Decoder_Read(
          &decoder,
          ETC_VEDC_Decoder_ReadType_CudaArrayFrame,
          &Frame
        ) != ETC_VEDC_Decoder_Error_OK) {
          return 0;
        }

        FrameSize = fan::vec2ui(ImageProperties.SizeX, ImageProperties.SizeY);

        ETC_VEDC_Decoder_ReadClear(&decoder, ETC_VEDC_Decoder_ReadType_CudaArrayFrame, &Frame);
        return r;
      }
      static auto get_decoders() {
        return std::to_array(_ETC_VEDC_DecoderList);
      }

      void sleep_thread() {
        uint64_t OneFrameTime = 1000000000 / encoder.settings.InputFrameRate;
        uint64_t CTime = T_nowi();
        uint64_t TimeDiff = CTime - FrameProcessStartTime;
        if (TimeDiff > OneFrameTime) {
          FrameProcessStartTime = CTime;
        }
        else {
          uint64_t SleepTime = OneFrameTime - TimeDiff;
          FrameProcessStartTime = CTime + SleepTime;
          TH_sleepi(SleepTime);
        }
      }

      MD_SCR_t mdscr;
      uint8_t* screen_buffer = 0;
      bool screen_read() {
        screen_buffer = MD_SCR_read(&mdscr);
        return screen_buffer != 0;
      }

      uint64_t EncoderStartTime = T_nowi();
      uint64_t FrameProcessStartTime = EncoderStartTime;
    private:
      fan::vec2 FrameRenderSize;
      fan::vec2ui FrameSize = 1;

      union ReadMethodData_t {
        ReadMethodData_t() {};
        ~ReadMethodData_t() {};
#ifdef __GPU_CUDA
        struct CudaArrayFrame_t {
          loco_t::cuda_textures_t cuda_textures;
        }CudaArrayFrame;
#endif
      }ReadMethodData;

      ETC_VEDC_Decoder_ReadType LastReadMethod = ETC_VEDC_Decoder_ReadType_Unknown;
      void NewReadMethod(loco_t::shape_t& universal_image_renderer, ETC_VEDC_Decoder_ReadType Type) {
        if (LastReadMethod == Type) {
          return;
        }
        switch (LastReadMethod) {
#ifdef __GPU_CUDA
        case ETC_VEDC_Decoder_ReadType_CudaArrayFrame: {
          ReadMethodData.CudaArrayFrame.cuda_textures.close(gloco, universal_image_renderer);
          break;
        }
#endif
        default: { break; }
        }
        switch (Type) {
#ifdef __GPU_CUDA
        case ETC_VEDC_Decoder_ReadType_CudaArrayFrame: {
          new (&ReadMethodData.CudaArrayFrame) ReadMethodData_t::CudaArrayFrame_t;
          break;
        }
#endif
        default: { break; }
        }
        LastReadMethod = Type;
      }
    };
  }
}