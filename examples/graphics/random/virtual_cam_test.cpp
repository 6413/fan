#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <softcam/softcam.h>

#include <olectl.h>
#include <initguid.h>

#include <softcamcore/DShowSoftcam.h>
#include <softcamcore/SenderAPI.h>

#undef min
#undef max

import fan;

// {AEF3B972-5FA5-4647-9571-358EB472BC9E}
DEFINE_GUID(CLSID_DShowSoftcam,
0xaef3b972, 0x5fa5, 0x4647, 0x95, 0x71, 0x35, 0x8e, 0xb4, 0x72, 0xbc, 0x9e);

namespace {
  const wchar_t FILTER_NAME[] = L"fan virtual camera";
  const GUID& FILTER_CLASSID = CLSID_DShowSoftcam;

  const AMOVIESETUP_MEDIATYPE s_pin_types[] = {
    { &MEDIATYPE_Video, &MEDIASUBTYPE_NULL }
  };

  const AMOVIESETUP_PIN s_pins[] = {
    {
      const_cast<LPWSTR>(L"Output"),
      FALSE, TRUE, FALSE, FALSE,
      &CLSID_NULL, NULL,
      1, s_pin_types
    }
  };

  const REGFILTER2 s_reg_filter2 = { 1, MERIT_DO_NOT_USE, 1, s_pins };

  CUnknown* WINAPI CreateSoftcamInstance(LPUNKNOWN lpunk, HRESULT* phr) {
    return softcam::Softcam::CreateInstance(lpunk, FILTER_CLASSID, phr);
  }
} // namespace

CFactoryTemplate g_Templates[] = {
  { FILTER_NAME, &FILTER_CLASSID, &CreateSoftcamInstance, NULL, nullptr }
};
int g_cTemplates = sizeof(g_Templates) / sizeof(g_Templates[0]);

STDAPI DllRegisterServer() {
  HRESULT hr = AMovieDllRegisterServer2(TRUE);
  if (FAILED(hr)) {
    return hr;
  }
  hr = CoInitialize(nullptr);
  if (FAILED(hr)) {
    return hr;
  }
  do {
    IFilterMapper2* pFM2 = nullptr;
    hr = CoCreateInstance(
      CLSID_FilterMapper2, nullptr, CLSCTX_INPROC_SERVER,
      IID_IFilterMapper2, (void**)&pFM2);
    if (FAILED(hr)) {
      break;
    }
    pFM2->UnregisterFilter(&CLSID_VideoInputDeviceCategory, 0, FILTER_CLASSID);
    hr = pFM2->RegisterFilter(
      FILTER_CLASSID, FILTER_NAME, 0,
      &CLSID_VideoInputDeviceCategory, FILTER_NAME, &s_reg_filter2);
    pFM2->Release();
  } while (0);
  CoFreeUnusedLibraries();
  CoUninitialize();
  return hr;
}

STDAPI DllUnregisterServer() {
  HRESULT hr = AMovieDllRegisterServer2(FALSE);
  if (FAILED(hr)) {
    return hr;
  }
  hr = CoInitialize(nullptr);
  if (FAILED(hr)) {
    return hr;
  }
  do {
    IFilterMapper2* pFM2 = nullptr;
    hr = CoCreateInstance(
      CLSID_FilterMapper2, nullptr, CLSCTX_INPROC_SERVER,
      IID_IFilterMapper2, (void**)&pFM2);
    if (FAILED(hr)) {
      break;
    }
    hr = pFM2->UnregisterFilter(
      &CLSID_VideoInputDeviceCategory, FILTER_NAME, FILTER_CLASSID);
    pFM2->Release();
  } while (0);
  CoFreeUnusedLibraries();
  CoUninitialize();
  return hr;
}

extern "C" BOOL WINAPI DllEntryPoint(HINSTANCE, ULONG, LPVOID);

BOOL APIENTRY DllMain(HANDLE hModule, DWORD dwReason, LPVOID lpReserved) {
  return DllEntryPoint((HINSTANCE)(hModule), dwReason, lpReserved);
}

extern "C" scCamera scCreateCamera(int width, int height, f32_t framerate) {
  return softcam::sender::CreateCamera(width, height, framerate);
}
extern "C" void scDeleteCamera(scCamera camera) {
  return softcam::sender::DeleteCamera(camera);
}
extern "C" void scSendFrame(scCamera camera, const void* image_bits) {
  return softcam::sender::SendFrame(camera, image_bits);
}
extern "C" bool scWaitForConnection(scCamera camera, f32_t timeout) {
  return softcam::sender::WaitForConnection(camera, timeout);
}

int main() {
  _putenv_s("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0");

  static constexpr int cam_width = 1920;
  static constexpr int cam_height = 1080;
  static constexpr fan::vec2 cam_size = fan::vec2(cam_width, cam_height);

  auto t0 = std::chrono::high_resolution_clock::now();
  auto print_elapsed = [&](const char* label) {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - t0).count();
    fan::print(label, ms, "ms");
  };

  print_elapsed("start");
  loco_t loco{ {.window_size = cam_size} };
  loco.window.set_size(cam_size);
  print_elapsed("loco init done");

  fan::image::info_t ii;
  ii.size = cam_size;
  ii.data = 0;

  fan::graphics::image_load_properties_t lp;
  lp.internal_format = fan::graphics::image_format_e::rgb_unorm;
  lp.format = fan::graphics::image_format_e::bgr_unorm;

  fan::graphics::image_t image = loco.image_load(ii, lp);
  print_elapsed("image loaded");

  fan::graphics::sprite_t s{{
    .position = fan::vec3(cam_size / 2.f, 100),
    .size = cam_size / 2.f,
    .image = image,
    .blending = true
  }};

  loco.window.add_resize_callback([&](const auto& d) {
    loco.viewport_set(
      gloco()->orthographic_render_view.viewport,
      fan::vec2(0, 0),
      d.size
    );
  });

  loco.set_vsync(false);

  scCamera cam = scCreateCamera(cam_width, cam_height, 30);
  print_elapsed("softcam created");

  auto walls = fan::graphics::physics::create_stroked_rectangle(cam_size / 2, cam_size / 4, 10.f);
  loco.get_physics_context().set_gravity(0);

  std::atomic<bool> capture_ready = false;
  std::atomic<bool> app_running = true;
  std::mutex frame_mutex;
  cv::Mat current_frame;
  cv::VideoCapture capture;

  std::thread capture_thread([&] {
    capture.open(0, cv::CAP_MSMF);
    if (!capture.isOpened()) {
      fan::print("error: could not open camera");
      return;
    }

    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(cv::CAP_PROP_FRAME_WIDTH,   cam_width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT,  cam_height);
    capture.set(cv::CAP_PROP_FPS,           30.0);

    capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
    capture.set(cv::CAP_PROP_AUTO_WB,       0);
    capture.set(cv::CAP_PROP_EXPOSURE,      -6.0);
    capture.set(cv::CAP_PROP_GAIN,          0.0);
    capture.set(cv::CAP_PROP_SHARPNESS,     0.0);
    capture_ready = true;

    while (app_running) {
      cv::Mat temp;
      capture >> temp;
      if (!temp.empty()) {
        std::lock_guard<std::mutex> lock(frame_mutex);
        current_frame = temp.clone();
      }
    }
  });

  std::mutex out_mutex;
  std::vector<uint8_t> out_pixel_data;
  std::atomic<bool> out_ready = false;

  std::thread sender_thread([&] {
    while (app_running) {
      if (out_ready) {
        std::vector<uint8_t> local_data;
        {
          std::lock_guard<std::mutex> lock(out_mutex);
          local_data = std::move(out_pixel_data);
          out_ready = false;
        }

        if (local_data.empty()) continue;

        cv::Mat image_rgba(cam_height, cam_width, CV_8UC4, (void*)local_data.data());
        cv::Mat image_bgr, flipped_image;
        cv::cvtColor(image_rgba, image_bgr, cv::COLOR_RGBA2BGR);
        cv::flip(image_bgr, flipped_image, 0);

        scSendFrame(cam, flipped_image.data);
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  });

  f32_t digital_gain    = 1.0f;
  f32_t detail_strength = 1.0f;
  f32_t black_floor     = 0.02f;
  f32_t noise_floor     = 0.0f;
  f32_t saturation      = 0.3f;
  f32_t bloom_str       = 0.01f;
  f32_t exposure_val    = 1.3f;
  f32_t contrast_val    = 0.9f;
  f32_t gamma_val       = 0.85f;
  
  f32_t key_color[3]    = { 0.5f, 0.5f, 0.5f };
  f32_t key_tolerance   = 0.15f;

  f64_t hw_exposure   = -6.0;
  f64_t hw_gain       = 0.0;
  f64_t hw_wb_temp    = 4600.0;
  f64_t hw_brightness = 128.0;
  f64_t hw_sharpness  = 0.0;

  fan::time::timer camera_fps;
  camera_fps.start(16.66e+6);

  print_elapsed("entering loop");
  loco.loop([&] {
    fan::graphics::gui::begin("shader");
    if (fan::graphics::gui::drag("digital_gain",    &digital_gain,    0.5f,   0.f,   80.f))
      loco.set_post_process("digital_gain",    digital_gain);
    if (fan::graphics::gui::drag("detail_strength", &detail_strength, 0.1f,   0.f,   20.f))
      loco.set_post_process("detail_strength", detail_strength);
    if (fan::graphics::gui::drag("black_floor",     &black_floor,     0.001f, 0.f,   0.1f))
      loco.set_post_process("black_floor",     black_floor);
    if (fan::graphics::gui::drag("noise_floor",     &noise_floor,     0.001f, 0.f,   0.05f))
      loco.set_post_process("noise_floor",     noise_floor);
    if (fan::graphics::gui::drag("saturation",      &saturation,      0.01f,  0.f,   1.f))
      loco.set_post_process("saturation",      saturation);
    if (fan::graphics::gui::drag("bloom_strength",  &bloom_str,       0.005f, 0.f,   0.2f))
      loco.set_post_process("bloom_strength",  bloom_str);
    if (fan::graphics::gui::drag("exposure",        &exposure_val,    0.05f,  0.1f,  5.f))
      loco.set_post_process("exposure",        exposure_val);
    if (fan::graphics::gui::drag("contrast",        &contrast_val,    0.01f,  0.5f,  2.f))
      loco.set_post_process("contrast",        contrast_val);
    if (fan::graphics::gui::drag("gamma",           &gamma_val,       0.01f,  0.3f,  2.f))
      loco.set_post_process("gamma",           gamma_val);
    
    fan::graphics::gui::separator();
    if (fan::graphics::gui::color_edit3("key_color", (fan::vec3*)key_color))
      loco.set_post_process("key_color", fan::vec3(key_color[0], key_color[1], key_color[2]));
    if (fan::graphics::gui::drag("key_tolerance",   &key_tolerance,   0.005f, 0.f,   1.f))
      loco.set_post_process("key_tolerance",   key_tolerance);
    fan::graphics::gui::end();

    fan::graphics::gui::begin("camera");
    if (capture_ready) {
      if (fan::graphics::gui::drag("hw_exposure",   &hw_exposure,   0.5,   -13.0, 0.0))
        capture.set(cv::CAP_PROP_EXPOSURE,       hw_exposure);
      if (fan::graphics::gui::drag("hw_gain",       &hw_gain,       1.0,    0.0,  100.0))
        capture.set(cv::CAP_PROP_GAIN,           hw_gain);
      if (fan::graphics::gui::drag("hw_wb_temp",    &hw_wb_temp,  100.0, 2000.0, 8000.0))
        capture.set(cv::CAP_PROP_WB_TEMPERATURE, hw_wb_temp);
      if (fan::graphics::gui::drag("hw_brightness", &hw_brightness, 1.0,    0.0,  255.0))
        capture.set(cv::CAP_PROP_BRIGHTNESS,     hw_brightness);
      if (fan::graphics::gui::drag("hw_sharpness",  &hw_sharpness,  1.0,    0.0,  255.0))
        capture.set(cv::CAP_PROP_SHARPNESS,      hw_sharpness);
    }
    else {
      fan::graphics::gui::text("camera initializing...");
    }
    fan::graphics::gui::end();

    if (!capture_ready) {
      return;
    }

    if (camera_fps.finished()) {
      cv::Mat frame;
      {
        std::lock_guard<std::mutex> lock(frame_mutex);
        if (current_frame.empty()) return;
        cv::swap(frame, current_frame);
      }

      ii.data = frame.data;
      ii.size = fan::vec2(frame.cols, frame.rows);
      loco.image_unload(image);
      image = loco.image_load(ii, lp);

      auto pixel_data = loco.image_get_pixel_data(loco.get_color_buffer(0), fan::graphics::image_format_e::rgba_unorm);
      
      {
        std::lock_guard<std::mutex> lock(out_mutex);
        out_pixel_data = std::move(pixel_data);
        out_ready = true;
      }

      camera_fps.restart();
    }
  });

  app_running = false;
  if (capture_thread.joinable()) {
    capture_thread.join();
  }
  if (sender_thread.joinable()) {
    sender_thread.join();
  }
  scDeleteCamera(cam);
}
