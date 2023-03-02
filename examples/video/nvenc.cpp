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
#define loco_yuv420p
#define loco_pixel_format_renderer
#include _FAN_PATH(graphics/loco.h)

#include _FAN_PATH(video/nvec_cuda.h)

struct pile_t {

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {

    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_camera(
      &camera,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
    //fan::vec2 ratio = window_size / window_size.max();
    //std::swap(ratio.x, ratio.y);
    //camera.set_ortho(
    //  ortho_x * ratio.x, 
    //  ortho_y * ratio.y
    //);
    viewport.set(loco.get_context(), 0, d.size, d.size);
      });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);
  }

  loco_t loco;
  loco_t::camera_t camera;
  fan::graphics::viewport_t viewport;
  fan::graphics::cid_t cid[5];
};

pile_t* pile = new pile_t;

#include _FAN_PATH(video/nvdec.h)

fan::string EncodeCuda(CUcontext cuContext, const fan::string& data, const fan::vec2ui& size, NV_ENC_BUFFER_FORMAT eFormat) {

  NvEncoderCuda enc(cuContext, size.x, size.y, eFormat);

  NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
  NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
  initializeParams.encodeConfig = &encodeConfig;
  enc.CreateDefaultEncoderParams(&initializeParams, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_LOW_LATENCY_HP_GUID);

  enc.CreateEncoder(&initializeParams);

  int nFrameSize = enc.GetFrameSize();

  fan::string bytes;
  uint64_t offset = 0;

  while (offset + nFrameSize < data.size()) {

    std::vector<std::vector<uint8_t>> vPacket;
    if (!(offset % nFrameSize)) {
      const NvEncInputFrame* encoderInputFrame = enc.GetNextInputFrame();
      NvEncoderCuda::CopyToDeviceFrame(cuContext, (void*)&data[offset], 0, (CUdeviceptr)encoderInputFrame->inputPtr,
        (int)encoderInputFrame->pitch,
        enc.GetEncodeWidth(),
        enc.GetEncodeHeight(),
        CU_MEMORYTYPE_HOST,
        encoderInputFrame->bufferFormat,
        encoderInputFrame->chromaOffsets,
        encoderInputFrame->numChromaPlanes);

      enc.EncodeFrame(vPacket);
    }

    offset += nFrameSize;

    if (offset + nFrameSize >= data.size()){
      enc.EndEncode(vPacket);
    }
    for (const auto& packet : vPacket) {
      bytes += fan::string(packet.data(), packet.data() + packet.size());
    }
  }

  enc.DestroyEncoder();
  return bytes;
}

int main() {
  fan::vec2 encode_size(1280, 720);

  NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_ARGB;
  int iGpu = 0;

  fan::string video_data;

  try {
    if (!__GPU_IS_CUDA_INITED) {
      fan::cuda::check_error(cuInit(0));
      __GPU_IS_CUDA_INITED = true;
    }
    int nGpu = 0;
    fan::cuda::check_error(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu)
    {
      std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
      return 1;
    }
    CUdevice cuDevice = 0;
    fan::cuda::check_error(cuDeviceGet(&cuDevice, iGpu));
    char szDeviceName[80];
    fan::cuda::check_error(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    CUcontext cuContext = NULL;
    fan::cuda::check_error(cuCtxCreate(&cuContext, 0, cuDevice));

    fan::string data;
    fan::io::file::read("o.brgx", &data);

    video_data = EncodeCuda(cuContext, data, encode_size, eFormat);
  }
  catch (const std::exception& ex) {
    std::cout << ex.what();
    return 1;
  }

  pile->loco.set_vsync(false);

  fan::cuda::nv_decoder_t nv(&pile->loco);

  loco_t::pixel_format_renderer_t::properties_t p;

  p.camera = &pile->camera;
  p.viewport = &pile->viewport;
  p.size = 1;
  p.images[0] = &nv.image_y;
  p.images[1] = &nv.image_vu;
  p.pixel_format = fan::pixel_format::nv12;
  pile->loco.pixel_format_renderer.push_back(&pile->cid[1], p);

  nv.start_decoding(video_data);
  //pile->loco.loop([] {});

};