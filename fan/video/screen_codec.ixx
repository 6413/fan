module;

#include <fan/types/types.h>
#include <WITCH/WITCH.h>
#define ETC_VEDC_Encode_DefineEncoder_OpenH264
#define ETC_VEDC_Encode_DefineEncoder_x264
//#if defined(__platform_windows)
#define ETC_VEDC_Encode_DefineEncoder_nvenc
//#endif

#define ETC_VEDC_Decoder_DefineCodec_OpenH264
//#if defined(__platform_windows)
#define ETC_VEDC_Decoder_DefineCodec_cuvid
//#endif#


#include <mutex>

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
import fan.fmt;

export namespace fan {
  namespace graphics {
    struct codec_update_e {
      enum {
        codec = 1 << 0,
        rate_control = 1 << 1,
        frame_rate = 1 << 2
      };
    };
    struct screen_encode_t : ETC_VEDC_Encode_t{

      void close_encoder() {
        ETC_VEDC_Encode_Close(this);
      }
      void open_encoder() {
        if (name == "Nothing") {
          ETC_VEDC_Encode_OpenNothing(this);
          return;
        }
        ETC_VEDC_Encode_Error err = ETC_VEDC_Encode_Open(
          this,
          name.size(),
          name.c_str(),
          &settings,
          NULL);
        if (err != ETC_VEDC_Encode_Error_Success) {
          fan::print("failed to open encoder ", err);
          /* TODO */
          __abort();
        }
      }

      screen_encode_t() {

        if (MD_SCR_open(&mdscr) != 0) {
          __abort();
        }

        settings = {
         .CodecStandard = ETC_VCODECSTD_H264,
         .UsageType = ETC_VEDC_EncoderSetting_UsageType_Realtime,
         .RateControl{
           .Type = ETC_VEDC_EncoderSetting_RateControlType_VBR,
           .VBR = {.bps = 16000000 }
         },
         .InputFrameRate = 30
        };

        settings.FrameSizeX = mdscr.Geometry.Resolution.x;
        settings.FrameSizeY = mdscr.Geometry.Resolution.y;

        open_encoder();
      }

      std::string name = "x264";
      uintptr_t new_codec = 0;
      
      ETC_VEDC_EncoderSetting_t settings;
      void* data{};
      uintptr_t amount = 0;
      uint8_t update_flags = 0;

      // returns if is readable
      bool encode_write(f32_t FrameProcessStartTime) {
        if (
          settings.FrameSizeX != mdscr.Geometry.Resolution.x ||
          settings.FrameSizeY != mdscr.Geometry.Resolution.y
          ) {
          settings.FrameSizeX = mdscr.Geometry.Resolution.x;
          settings.FrameSizeY = mdscr.Geometry.Resolution.y;
          sint32_t err = ETC_VEDC_Encode_SetFrameSize(
            this,
            settings.FrameSizeX,
            settings.FrameSizeY);
          if (err != 0) {
            /* TODO */
            __abort();
          }
        }

        settings.FrameSizeX = mdscr.Geometry.Resolution.x;
        settings.FrameSizeY = mdscr.Geometry.Resolution.y;

        if (update_flags & codec_update_e::codec) {
          mutex.lock();
          close_encoder();
          EncoderID = new_codec;
          name = get_encoders()[EncoderID].Name;
          open_encoder();
          update_flags = 0;
          mutex.unlock();
        }
        else {
          if (update_flags & codec_update_e::rate_control) {
            update_flags &= ~codec_update_e::rate_control;
            ETC_VEDC_Encode_Error err = ETC_VEDC_Encode_SetRateControl(
              this,
              &settings.RateControl);
            if (err != ETC_VEDC_Encode_Error_Success) {
              /* TODO */
              __abort();
            }
          }
          if (update_flags & codec_update_e::frame_rate) {
            update_flags &= ~codec_update_e::frame_rate;
            ETC_VEDC_Encode_Error err = ETC_VEDC_Encode_SetInputFrameRate(
              this,
              settings.InputFrameRate);
            if (err != ETC_VEDC_Encode_Error_Success) {
              /* TODO */
              __abort();
            }
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


          ETC_VEDC_Encode_Error err = ETC_VEDC_Encode_Write(this, ETC_VEDC_Encode_WriteType_Frame, &Frame);
          if (err != ETC_VEDC_Encode_Error_Success) {
            /* TODO */
            __abort();
          }
        }

        return ETC_VEDC_Encode_IsReadable(this) != false;
      }
      uintptr_t encode_read() {
        ETC_VEDC_Encode_PacketInfo PacketInfo;
        amount = ETC_VEDC_Encode_Read(this, &PacketInfo, &data);
        return amount;
      }
      static auto get_encoders() {
        return std::to_array(_ETC_VEDC_EncoderList);
      }

      void init(f32_t encode_start) {
        EncoderStartTime = encode_start;
      }

      MD_SCR_t mdscr;
      uint8_t* screen_buffer = 0;
      bool screen_read() {
        screen_buffer = MD_SCR_read(&mdscr);
        return screen_buffer != 0;
      }

      uint64_t EncoderStartTime = T_nowi();
      std::mutex mutex;
    };

    struct screen_decode_t : ETC_VEDC_Decoder_t{

      void open_decoder() {
        if (name == "Nothing") {
          ETC_VEDC_Decoder_OpenNothing(this);
          return;
        }
        auto r = ETC_VEDC_Decoder_Open(
          this,
          name.size(),
          name.c_str(),
          0);
        if (r != ETC_VEDC_Decoder_Error_OK) {


          fan::printn8(
            "[CLIENT] [WARNING] [DECODER] ", __FUNCTION__, " ", __FILE__, ":", __LINE__,
            " (ETC_VEDC_Decoder_Open returned (", r, ") for encoder \"",
            name, "\""
          );

          fan::printn8(
            "[CLIENT] [WARNING] [DECODER] ", __FUNCTION__, " ", __FILE__, ":", __LINE__,
            " falling back to OpenH264 decoder."
          );
        }
      }
      void close_decoder() {
        ETC_VEDC_Decoder_Close(this);
      }

      screen_decode_t() {
        open_decoder();
      }

      void init(f32_t EncoderStartTime) {
        FrameProcessStartTime = EncoderStartTime;
      }

      uintptr_t decode(void* data, uintptr_t length, loco_t::shape_t& universal_image_renderer) {
        if (update_flags & codec_update_e::codec) {
          bool lock_success = mutex.try_lock();

#if defined(ETC_VEDC_Decoder_DefineCodec_cuvid)
          if (name == "cuvid") { // is there better way?
            // close resources before closing cu context
            ReadMethodData.CudaArrayFrame.cuda_textures.close(gloco, universal_image_renderer);
          }
#endif

          close_decoder();
          DecoderID = new_codec;
          name = get_decoders()[DecoderID].Name;
          open_decoder();
          update_flags = 0;
          if (lock_success) {
            mutex.unlock();
          }
        }

        sintptr_t r = ETC_VEDC_Decoder_Write(this, (uint8_t*)data, length);

        if (!ETC_VEDC_Decoder_IsReadable(this)) {
          return 0;
        }
        
        if (ETC_VEDC_Decoder_IsReadType(this, ETC_VEDC_Decoder_ReadType_CudaArrayFrame)) {
          NewReadMethod(universal_image_renderer, ETC_VEDC_Decoder_ReadType_CudaArrayFrame);
          if (cuCtxSetCurrent((CUcontext)ETC_VEDC_Decoder_GetUnique(this, ETC_VEDC_Decoder_UniqueType_CudaContext)) != CUDA_SUCCESS) {
            __abort();
          }

          ETC_VEDC_Decoder_ImageProperties_t props;
          ETC_VEDC_Decoder_GetReadImageProperties(this, ETC_VEDC_Decoder_ReadType_CudaArrayFrame, &props);
          ReadMethodData.CudaArrayFrame.cuda_textures.resize(gloco, universal_image_renderer, fan::graphics::image_format::nv12, fan::vec2ui(props.SizeX, props.SizeY));

          ETC_VEDC_Decoder_CudaArrayFrame_t Frame;
          for (uint32_t i = 0; i < 4; i++) {
            Frame.Array[i] = ReadMethodData.CudaArrayFrame.cuda_textures.get_array(i);
          }

          if (ETC_VEDC_Decoder_Read(this, ETC_VEDC_Decoder_ReadType_CudaArrayFrame, &Frame) != ETC_VEDC_Decoder_Error_OK) {
            return 0;
          }
          FrameSize = fan::vec2ui(props.SizeX, props.SizeY);
          ETC_VEDC_Decoder_ReadClear(this, ETC_VEDC_Decoder_ReadType_CudaArrayFrame, &Frame);
        }
        if (ETC_VEDC_Decoder_IsReadType(this, ETC_VEDC_Decoder_ReadType_Frame)) {
          NewReadMethod(universal_image_renderer, ETC_VEDC_Decoder_ReadType_Frame);
          ETC_VEDC_Decoder_Frame_t Frame;
          if (ETC_VEDC_Decoder_Read(this, ETC_VEDC_Decoder_ReadType_Frame, &Frame) != ETC_VEDC_Decoder_Error_OK) {
            return 0;
          }
          if (Frame.Properties.Stride[0] != Frame.Properties.Stride[1] * 2) {
            fan::print_format("[CLIENT] [WARNING] {} {}:{} fan doesnt support strides {} {}", __FUNCTION__, __FILE__, __LINE__, Frame.Properties.Stride[0], Frame.Properties.Stride[1]);
            __abort();
          }

          uint32_t pixel_format;
          if (Frame.Properties.PixelFormat == PIXF_YUV420p) {
            pixel_format = fan::graphics::image_format::yuv420p;
          }
          else if (Frame.Properties.PixelFormat == PIXF_YUVNV12) {
            pixel_format = fan::graphics::image_format::nv12;
          }
          else {
            __abort();
          }

          f32_t sx = (f32_t)Frame.Properties.SizeX / Frame.Properties.Stride[0];
          universal_image_renderer.set_tc_size(fan::vec2(sx, 1));
          universal_image_renderer.reload(pixel_format, (void**)Frame.Data, fan::vec2ui(Frame.Properties.Stride[0], Frame.Properties.SizeY));
          FrameSize = fan::vec2ui(Frame.Properties.SizeX, Frame.Properties.SizeY);
          ETC_VEDC_Decoder_ReadClear(this, ETC_VEDC_Decoder_ReadType_Frame, &Frame);
        }

        return r;
      }
      static auto get_decoders() {
        return std::to_array(_ETC_VEDC_DecoderList);
      }

      void sleep_thread(f32_t frame_rate) {
        uint64_t OneFrameTime = 1000000000 / frame_rate;
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

      std::string name = "cuvid";
      uintptr_t new_codec = 0;

      uint64_t FrameProcessStartTime = 0;

      fan::vec2 FrameRenderSize;
      fan::vec2ui FrameSize = 1;
      uint8_t update_flags = 0;
      std::mutex mutex;

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
        switch (Type) {
#ifdef __GPU_CUDA
        case ETC_VEDC_Decoder_ReadType_CudaArrayFrame: {
          new (&ReadMethodData.CudaArrayFrame) ReadMethodData_t::CudaArrayFrame_t;
          universal_image_renderer.set_tc_size(fan::vec2(1, 1));
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