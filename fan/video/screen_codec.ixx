module;

#include <fan/types/types.h>
#include <fan/time/timer.h>

#include <WITCH/WITCH.h>
#define ETC_VEDC_Encode_DefineEncoder_OpenH264
#define ETC_VEDC_Encode_DefineEncoder_x264
//#if defined(__platform_windows)
#if __has_include("cuda.h")
  #define ETC_VEDC_Encode_DefineEncoder_nvenc
#endif
//#endif

#define ETC_VEDC_Decoder_DefineCodec_OpenH264
//#if defined(__platform_windows)
#if __has_include("cuda.h")
  #define ETC_VEDC_Decoder_DefineCodec_cuvid
#endif
//#endif#


#include <string>
#include <mutex>
#include <array>
#include <cstring>
#include <atomic>
#include <functional>
#include <thread>
#include <chrono>

#include <WITCH/PR/PR.h>
#include <WITCH/MD/SCR/SCR.h>
//#include <WITCH/MD/Mice.h>
//#include <WITCH/MD/Keyboard/Keyboard.h>

//#include <WITCH/HASH/SHA.h>
#include <WITCH/ETC/VEDC/Encode.h>
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
        frame_rate = 1 << 2,
      };
      enum {
        reset_IDR = ETC_VEDC_EncoderFlag_ResetIDR
      };
    };
    struct screen_encode_t : ETC_VEDC_Encode_t {

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
        std::memset(&mdscr, 0, sizeof(mdscr));
        if (int ret = MD_SCR_open(&mdscr); ret != 0) {
          fan::print("failed to open screen:" + std::to_string(ret));
          __abort();
        }

        settings = {
         .CodecStandard = ETC_VCODECSTD_H264,
         .UsageType = ETC_VEDC_EncoderSetting_UsageType_Realtime,
         .RateControl{
           .Type = ETC_VEDC_EncoderSetting_RateControlType_VBR,
           .VBR = {.bps = 10000000 }
         },
         .InputFrameRate = 30
        };

        settings.FrameSizeX = mdscr.Geometry.Resolution.x;
        settings.FrameSizeY = mdscr.Geometry.Resolution.y;

        open_encoder();
      }
      static auto get_encoders() {
        return std::to_array(_ETC_VEDC_EncoderList);
      }

      std::string name = "x264";
      uintptr_t new_codec = 0;

      ETC_VEDC_EncoderSetting_t settings;
      void* data{};
      uintptr_t amount = 0;
      uint8_t update_flags = 0;

      // returns if is readable
      bool encode_write() {
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


          ETC_VEDC_Encode_Error err = ETC_VEDC_Encode_Write(this, ETC_VEDC_Encode_WriteType_Frame, &Frame, encode_write_flags);
          if (err != ETC_VEDC_Encode_Error_Success) {
            /* TODO */
            __abort();
          }
          if (encode_write_flags & ETC_VEDC_EncoderFlag_ResetIDR) {
            encode_write_flags &= ~ETC_VEDC_EncoderFlag_ResetIDR;
          }
        }

        return ETC_VEDC_Encode_IsReadable(this) != false;
      }
      uintptr_t encode_read() {
        ETC_VEDC_Encode_PacketInfo PacketInfo;
        amount = ETC_VEDC_Encode_Read(this, &PacketInfo, &data);
        return amount;
      }

      void sleep_thread() {
        uint64_t OneFrameTime = 1000000000 / settings.InputFrameRate;
        uint64_t CTime = fan::time::clock::now();
        uint64_t TimeDiff = CTime - FrameProcessStartTime;
        if (TimeDiff > OneFrameTime) {
          FrameProcessStartTime = CTime;
        }
        else {
          uint64_t SleepTime = OneFrameTime - TimeDiff;
          FrameProcessStartTime = CTime + SleepTime;
          fan::event::sleep(SleepTime / 1000000); // todo bad
        }
      }

      uint8_t encode_write_flags = 0;
      uint64_t EncoderStartTime = fan::time::clock::now();
      uint64_t FrameProcessStartTime = EncoderStartTime;
      MD_SCR_t mdscr;

      uint8_t* screen_buffer = 0;
      bool screen_read() {
        screen_buffer = MD_SCR_read(&mdscr);
        return screen_buffer != 0;
      }

      std::timed_mutex mutex;
    };

    struct screen_decode_t : ETC_VEDC_Decoder_t {

      void open_decoder() {
        if (name == "Nothing") {
          ETC_VEDC_Decoder_OpenNothing(this);
          return;
        }

        auto r = ETC_VEDC_Decoder_Open(
          this,
          name.size(),
          name.c_str(),
          0
        );
        if (r != ETC_VEDC_Decoder_Error_OK) {
          fan::printn8("\n[CLIENT] [WARNING] [DECODER] ", __FUNCTION__, " ", __FILE__, ":", __LINE__,
            " (ETC_VEDC_Decoder_Open returned (", r, ") for encoder \"", name, "\"", "\n"
          );

          fan::printn8("\n[WARNING] [DECODER] ", __FUNCTION__, " ", __FILE__, ":", __LINE__, "\n");
        }
        closed = false;
      }
      void close_decoder() {
        ETC_VEDC_Decoder_Close(this);
        closed = true;
      }

      screen_decode_t() {
        open_decoder();
      }

      static auto get_decoders() {
        return std::to_array(_ETC_VEDC_DecoderList);
      }

      struct decode_data_t {
        std::array<std::vector<uint8_t>, 4> data;
        std::array<fan::vec2ui, 4> stride;
        fan::vec2ui image_size = 0;
        uint8_t pixel_format = 0;
        uint8_t type = 0;
      };
      loco_t::shape_t* cached_universal_image_renderer = 0;

        decode_data_t decode(void* data, uintptr_t length, loco_t::shape_t & universal_image_renderer) {
          decode_data_t ret;

          cached_universal_image_renderer = &universal_image_renderer;

          {
            std::lock_guard<std::mutex> lock(mutex);

            if (update_flags & codec_update_e::codec) {
#if defined(ETC_VEDC_Decoder_DefineCodec_cuvid)
              if (name == "cuvid") {
                auto cleanup_lambda = [this]() {
                  try {
                    if (cached_universal_image_renderer) {
                      ReadMethodData.CudaArrayFrame.cuda_textures.close(
                        gloco,
                        *static_cast<loco_t::shape_t*>(cached_universal_image_renderer)
                      );
                    }
                  }
                  catch (...) {
                  }
                  };

                graphics_queue_callback(cleanup_lambda);
              }
#endif

              // Execute decoder change immediately (no waiting for graphics cleanup)
              close_decoder();
              DecoderID = new_codec;
              name = get_decoders()[DecoderID].Name;
              open_decoder();
              update_flags = 0;

              // Clear cached pointer and callback
              cached_universal_image_renderer = nullptr;

              // Return immediately with decoder changed signal
              ret.type = 253; // Special "decoder changed" signal
              return ret;
            }
          }
          if (closed) {
            ret.type = 254; // Special "decoder closed" signal
            return ret;
          }

          sintptr_t r = ETC_VEDC_Decoder_Write(this, (uint8_t*)data, length);

          if (!ETC_VEDC_Decoder_IsReadable(this)) {
            ret.type = 252; // Special "not readable" signal
            return ret;
          }

#if defined(ETC_VEDC_Decoder_DefineCodec_cuvid)
          if (ETC_VEDC_Decoder_IsReadType(this, ETC_VEDC_Decoder_ReadType_CudaArrayFrame)) {
            ret.type = 0;
          }
          else
#endif
            if (ETC_VEDC_Decoder_IsReadType(this, ETC_VEDC_Decoder_ReadType_Frame)) {
              ret.type = 1;
            }

          if (ETC_VEDC_Decoder_IsReadType(this, ETC_VEDC_Decoder_ReadType_Frame)) {
            NewReadMethod(universal_image_renderer, ETC_VEDC_Decoder_ReadType_Frame);
            ETC_VEDC_Decoder_Frame_t Frame;
            if (ETC_VEDC_Decoder_Read(this, ETC_VEDC_Decoder_ReadType_Frame, &Frame) != ETC_VEDC_Decoder_Error_OK) {
              ret.type = 251; // Special "read failed" signal
              return ret;
            }
            if (Frame.Properties.Stride[0] != Frame.Properties.Stride[1] * 2) {
              fan::print_format("[CLIENT] [WARNING] {} {}:{} fan doesnt support strides {} {}", __FUNCTION__, __FILE__, __LINE__, Frame.Properties.Stride[0], Frame.Properties.Stride[1]);
              ret.type = 250; // Special "unsupported stride" signal
              return ret;
            }

            uint32_t pixel_format;
            ret.image_size = { Frame.Properties.SizeX, Frame.Properties.SizeY };
            if (Frame.Properties.PixelFormat == PIXF_YUV420p) {
              pixel_format = fan::graphics::image_format::yuv420p;
              auto image_sizes = fan::graphics::get_image_sizes(pixel_format, fan::vec2ui(Frame.Properties.Stride[0], Frame.Properties.SizeY));
              ret.data[0] = std::vector<uint8_t>((uint8_t*)Frame.Data[0], (uint8_t*)Frame.Data[0] + image_sizes[0].multiply());
              ret.data[1] = std::vector<uint8_t>((uint8_t*)Frame.Data[1], (uint8_t*)Frame.Data[1] + image_sizes[1].multiply());
              ret.data[2] = std::vector<uint8_t>((uint8_t*)Frame.Data[2], (uint8_t*)Frame.Data[2] + image_sizes[2].multiply());
              std::memcpy(ret.stride.data(), Frame.Properties.Stride, sizeof(ret.stride[0]) * 3);
            }
            else if (Frame.Properties.PixelFormat == PIXF_YUVNV12) {
              pixel_format = fan::graphics::image_format::nv12;
              auto image_sizes = fan::graphics::get_image_sizes(pixel_format, fan::vec2ui(Frame.Properties.Stride[0], Frame.Properties.SizeY));
              ret.data[0] = std::vector<uint8_t>((uint8_t*)Frame.Data[0], (uint8_t*)Frame.Data[0] + image_sizes[0].multiply());
              ret.data[1] = std::vector<uint8_t>((uint8_t*)Frame.Data[1], (uint8_t*)Frame.Data[1] + image_sizes[1].multiply());
              std::memcpy(ret.stride.data(), Frame.Properties.Stride, sizeof(ret.stride[0]) * 2);
            }
            else {
              ret.type = 249; // Special "unsupported pixel format" signal
              return ret;
            }
            ret.pixel_format = pixel_format;
            FrameSize = fan::vec2ui(Frame.Properties.SizeX, Frame.Properties.SizeY);
            ETC_VEDC_Decoder_ReadClear(this, ETC_VEDC_Decoder_ReadType_Frame, &Frame);
          }

          return ret;
        }

        bool decode_cuvid(loco_t::shape_t & universal_image_renderer) {
#if defined(ETC_VEDC_Decoder_DefineCodec_cuvid)
          if (ETC_VEDC_Decoder_IsReadType(this, ETC_VEDC_Decoder_ReadType_CudaArrayFrame) == false) {
            return false;
          }

          if (cuCtxSetCurrent((CUcontext)ETC_VEDC_Decoder_GetUnique(this, ETC_VEDC_Decoder_UniqueType_CudaContext)) != CUDA_SUCCESS) {
            __abort();
          }
          NewReadMethod(universal_image_renderer, ETC_VEDC_Decoder_ReadType_CudaArrayFrame);

          ETC_VEDC_Decoder_ImageProperties_t props;
          ETC_VEDC_Decoder_GetReadImageProperties(this, ETC_VEDC_Decoder_ReadType_CudaArrayFrame, &props);
          try {
            ReadMethodData.CudaArrayFrame.cuda_textures.resize(gloco, universal_image_renderer, fan::graphics::image_format::nv12, fan::vec2ui(props.SizeX, props.SizeY));
          }
          catch (fan::exception_t e) {
            fan::print("cuda_texture.resize() failed:"_str + e.reason);
            ReadMethodData.CudaArrayFrame.cuda_textures.close(gloco, universal_image_renderer);
            return false;
          }

          ETC_VEDC_Decoder_CudaArrayFrame_t Frame;
          for (uint32_t i = 0; i < 4; i++) {
            Frame.Array[i] = ReadMethodData.CudaArrayFrame.cuda_textures.get_array(i);
          }

          if (ETC_VEDC_Decoder_Read(this, ETC_VEDC_Decoder_ReadType_CudaArrayFrame, &Frame) != ETC_VEDC_Decoder_Error_OK) {
            return false;
          }
          FrameSize = fan::vec2ui(props.SizeX, props.SizeY);
          ETC_VEDC_Decoder_ReadClear(this, ETC_VEDC_Decoder_ReadType_CudaArrayFrame, &Frame);
          return true;
#else
          return false;
#endif
        }

        void graphics_queue_callback(const std::function<void()> f) {
          graphics_queue.emplace_back(f);
        }

        bool closed = true;

        std::string name = "OpenH264";
        uintptr_t new_codec = 0;
        std::vector<std::function<void()>> graphics_queue;

        fan::vec2 FrameRenderSize;
        fan::vec2ui FrameSize = 1;
        uint8_t update_flags = 0;
        std::mutex mutex;
        std::mutex context_mutex;

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
        void NewReadMethod(loco_t::shape_t & universal_image_renderer, ETC_VEDC_Decoder_ReadType Type) {
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
    };
  }