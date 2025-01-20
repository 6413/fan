#include <fan/pch.h>

// defined witch
#undef EMPTY

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <softcam/softcam.h>

#include <olectl.h>
#include <initguid.h>

#include <softcamcore/DShowSoftcam.h>
#include <softcamcore/SenderAPI.h>


// {AEF3B972-5FA5-4647-9571-358EB472BC9E}
DEFINE_GUID(CLSID_DShowSoftcam,
0xaef3b972, 0x5fa5, 0x4647, 0x95, 0x71, 0x35, 0x8e, 0xb4, 0x72, 0xbc, 0x9e);


namespace {

  // Setup data

  const wchar_t FILTER_NAME[] = L"fan virtual camera";
  const GUID& FILTER_CLASSID = CLSID_DShowSoftcam;

  const AMOVIESETUP_MEDIATYPE s_pin_types[] =
  {
    {
      &MEDIATYPE_Video,       // Major type
      &MEDIASUBTYPE_NULL      // Minor type
    }
  };

  const AMOVIESETUP_PIN s_pins[] =
  {
    {
      const_cast<LPWSTR>(L"Output"),  // Pin string name
      FALSE,                  // Is it rendered
      TRUE,                   // Is it an output
      FALSE,                  // Can we have none
      FALSE,                  // Can we have many
      &CLSID_NULL,            // Connects to filter
      NULL,                   // Connects to pin
      1,                      // Number of types
      s_pin_types             // Pin details
    }
  };

  const REGFILTER2 s_reg_filter2 =
  {
    1,
    MERIT_DO_NOT_USE,
    1,
    s_pins
  };

  CUnknown* WINAPI CreateSoftcamInstance(LPUNKNOWN lpunk, HRESULT* phr)
  {
    return softcam::Softcam::CreateInstance(lpunk, FILTER_CLASSID, phr);
  }

} // namespace

// COM global table of objects in this dll

CFactoryTemplate g_Templates[] =
{
  {
    FILTER_NAME,
    &FILTER_CLASSID,
    &CreateSoftcamInstance,
    NULL,
    nullptr
  }
};
int g_cTemplates = sizeof(g_Templates) / sizeof(g_Templates[0]);


STDAPI DllRegisterServer()
{
  HRESULT hr = AMovieDllRegisterServer2(TRUE);
  if (FAILED(hr))
  {
    return hr;
  }
  hr = CoInitialize(nullptr);
  if (FAILED(hr))
  {
    return hr;
  }
  do
  {
    IFilterMapper2* pFM2 = nullptr;
    hr = CoCreateInstance(
            CLSID_FilterMapper2, nullptr, CLSCTX_INPROC_SERVER,
            IID_IFilterMapper2, (void**)&pFM2);
    if (FAILED(hr))
    {
      break;
    }
    pFM2->UnregisterFilter(
            &CLSID_VideoInputDeviceCategory,
            0,
            FILTER_CLASSID);
    hr = pFM2->RegisterFilter(
            FILTER_CLASSID,
            FILTER_NAME,
            0,
            &CLSID_VideoInputDeviceCategory,
            FILTER_NAME,
            &s_reg_filter2);
    pFM2->Release();
  } while (0);
  CoFreeUnusedLibraries();
  CoUninitialize();
  return hr;
}

STDAPI DllUnregisterServer()
{
  HRESULT hr = AMovieDllRegisterServer2(FALSE);
  if (FAILED(hr))
  {
    return hr;
  }
  hr = CoInitialize(nullptr);
  if (FAILED(hr))
  {
    return hr;
  }
  do
  {
    IFilterMapper2* pFM2 = nullptr;
    hr = CoCreateInstance(
            CLSID_FilterMapper2, nullptr, CLSCTX_INPROC_SERVER,
            IID_IFilterMapper2, (void**)&pFM2);
    if (FAILED(hr))
    {
      break;
    }
    hr = pFM2->UnregisterFilter(
            &CLSID_VideoInputDeviceCategory,
            FILTER_NAME,
            FILTER_CLASSID);
    pFM2->Release();
  } while (0);
  CoFreeUnusedLibraries();
  CoUninitialize();
  return hr;
}

extern "C" BOOL WINAPI DllEntryPoint(HINSTANCE, ULONG, LPVOID);

BOOL APIENTRY DllMain(HANDLE hModule, DWORD  dwReason, LPVOID lpReserved)
{
  return DllEntryPoint((HINSTANCE)(hModule), dwReason, lpReserved);
}


extern "C" scCamera scCreateCamera(int width, int height, float framerate)
{
  return softcam::sender::CreateCamera(width, height, framerate);
}

extern "C" void     scDeleteCamera(scCamera camera)
{
  return softcam::sender::DeleteCamera(camera);
}

extern "C" void     scSendFrame(scCamera camera, const void* image_bits)
{
  return softcam::sender::SendFrame(camera, image_bits);
}

extern "C" bool     scWaitForConnection(scCamera camera, float timeout)
{
  return softcam::sender::WaitForConnection(camera, timeout);
}

int main() {
  loco_t loco;


  fan::image::image_info_t ii;
  ii.size = fan::sys::get_screen_resolution();
  ii.data = 0;

  loco_t::image_t image;
  loco_t::image_t::load_properties_t lp;
  lp.internal_format = GL_RGB;
  lp.format = GL_BGR;
  lp.type = GL_UNSIGNED_BYTE;
  image.load(ii, lp);

  fan::graphics::sprite_t s{{
      .position = fan::vec3(0, 0, 100),
      .size = 1,
      .image = &image,
      .blending = true
    }};

  loco.window.add_resize_callback([&](const auto& d) {
    gloco->default_camera->viewport.set(fan::vec2(0, 0), d.size, d.size);
  });

  cv::VideoCapture capture(0);

  if (!capture.isOpened()) {
    fan::throw_error("Error: Could not open the camera.");
  }

  cv::Mat frame;

  loco.set_vsync(false);

  scCamera cam = scCreateCamera(800, 800, 30);

  fan::time::clock camera_fps;
  camera_fps.start(fan::time::nanoseconds(66.33e+6));
  cv::Mat flipped;

  uint8_t* pixels = new uint8_t[800 * 800 * 4];

  fan::graphics::text_t t{{
      .text = "hello from fan"
    }};

  fan::vec3 p = t.get_position();
  p.z = 10;
  t.set_position(p);


  static constexpr int wall_count = 4;
  fan::graphics::collider_static_t walls[wall_count];
  for (int i = 0; i < wall_count; ++i) {
    f32_t angle = 2 * fan::math::pi * i / wall_count;
    static constexpr int outline = 1.01;
    int x = std::cos(angle) / outline;
    int y = std::sin(angle) / outline;
    walls[i] = fan::graphics::rectangle_t{{
        .position = fan::vec2(x, y),
        .size = fan::vec2(
          std::abs(x) == 0 ? 1 : 0.1,
          std::abs(y) == 0 ? 1 : 0.1
        ),
      .color = fan::colors::red / 2
      }};
  }


  std::vector<fan::graphics::collider_dynamic_t> balls;
  static constexpr int ball_count = 30;
  balls.reserve(ball_count);
  for (int i = 0; i < ball_count; ++i) {
    balls.push_back(fan::graphics::letter_t{{
        .color = fan::random::color(),
        .position = fan::vec3(fan::random::vec2(-0.8, 0.8), 0),
        .font_size = 0.1,
        .letter_id = fan::random::string(1).get_utf8(0)
      }});
    balls.back().set_velocity(fan::random::vec2_direction(-1, 1) * 2);
  }

  fan::graphics::bcol.PreSolve_Shape_cb = [](
    bcol_t* bcol,
    const bcol_t::ShapeInfoPack_t* sip0,
    const bcol_t::ShapeInfoPack_t* sip1,
    bcol_t::Contact_Shape_t* Contact
    ) {
      if (sip0->ShapeEnum == sip1->ShapeEnum) {
        bcol->Contact_Shape_DisableContact(Contact);
        return;
      }

      fan::vec2 velocity = bcol->GetObject_Velocity(sip0->ObjectID);
      fan::vec2 p0 = bcol->GetObject_Position(sip0->ObjectID);
      fan::vec2 p1 = bcol->GetObject_Position(sip1->ObjectID);
      fan::vec2 wall_size = bcol->GetShape_Rectangle_Size(bcol->GetObjectExtraData(sip1->ObjectID)->shape_id);

      fan::vec2 reflection = fan::math::reflection_no_rot(velocity, p0, p1, wall_size);

      auto nr = gloco->m_write_queue.write_queue.NewNodeFirst();
      gloco->m_write_queue.write_queue[nr].cb = [oid = sip0->ObjectID, reflection] {
        fan::graphics::bcol.SetObject_Velocity(oid, reflection);
        };
    };

  f32_t angle = 0;
  loco.loop([&] {
    loco.get_fps();
    int idx = 0;
    for (auto& i : balls) {
      i.set_position(i.get_collider_position());
      i.set_angle(angle + idx);
    }
    angle += loco.get_delta_time();
    if (camera_fps.finished()) {
      do {
        capture >> frame;
        if (frame.empty()) {
          break;
        }
        ii.data = frame.data;
        ii.size = fan::vec2(frame.cols, frame.rows);


        //image.bind_texture();
        loco.color_buffers[0].bind_texture();
        //loco.color_buffers[0].get_texture()
        loco.get_context()->fan_opengl_call(GetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels));;

        cv::Mat image_rgb;

        cv::Mat image_rgba(800, 800, CV_8UC4, pixels);
        // Convert RGBA to RGB
        cv::cvtColor(image_rgba, image_rgb, cv::COLOR_RGBA2BGR);
        cv::Mat flipped_image;
        cv::flip(image_rgb, flipped_image, 0);

        scSendFrame(cam, flipped_image.data);

        image.unload();
        image.load(ii, lp);

      } while (0);
      camera_fps.restart();
    }
  });
  delete pixels;

}