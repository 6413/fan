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
  static constexpr int cam_width = 1920;
  static constexpr int cam_height = 1080;
  static constexpr fan::vec2 cam_size = fan::vec2(cam_width, cam_height);

  loco_t loco{ {.window_size = cam_size} };

  fan::image::info_t ii;
  ii.size = loco.window.get_current_monitor_resolution();
  ii.data = 0;

  fan::graphics::image_load_properties_t lp;
  lp.internal_format = fan::graphics::image_format_e::rgb_unorm;
  lp.format = fan::graphics::image_format_e::bgr_unorm;

  fan::graphics::image_t image = loco.image_load(ii, lp);

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

  cv::VideoCapture capture(0);
  if (!capture.isOpened()) {
    fan::throw_error("Error: Could not open the camera.");
  }
  capture.set(cv::CAP_PROP_FRAME_WIDTH, cam_size.x);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, cam_size.y);
  fan::print(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  loco.set_vsync(false);

  scCamera cam = scCreateCamera(cam_width, cam_height, 30);

  fan::time::timer camera_fps;
  camera_fps.start(66.33e+6);

  uint8_t* pixels = new uint8_t[cam_width * cam_height * 4];

  auto walls = fan::graphics::physics::create_stroked_rectangle(cam_size / 2, cam_size / 4, 10.f);

  loco.physics_context.set_gravity(0);

  loco.draw_end_cb.emplace_back([&] {
    auto pixel_data = loco.image_get_pixel_data(loco.gl.color_buffers[0], fan::graphics::image_format_e::rgba_unorm);
    std::memcpy(pixels, pixel_data.data(), pixel_data.size());
  });

  loco.update_physics(true);
  loco.loop([&] {
    if (camera_fps.finished()) {
      do {
        cv::Mat frame;
        capture >> frame;
        if (frame.empty()) {
          break;
        }
        ii.data = frame.data;
        ii.size = fan::vec2(frame.cols, frame.rows);

        cv::Mat image_rgb, flipped_image;
        cv::Mat image_rgba(cam_height, cam_width, CV_8UC4, pixels);
        cv::cvtColor(image_rgba, image_rgb, cv::COLOR_RGBA2BGR);

        cv::Mat resized_bgr;
        cv::resize(image_rgb, resized_bgr, cv::Size(cam_width, cam_height));
        cv::flip(resized_bgr, flipped_image, 0);

        scSendFrame(cam, flipped_image.data);

        loco.image_unload(image);
        image = loco.image_load(ii, lp);
      } while (0);
      camera_fps.restart();
    }
  });

  delete[] pixels;
}