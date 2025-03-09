#include <fan/pch.h>

struct gamma_ramp_t {
  WORD Red[256];
  WORD Green[256];
  WORD Blue[256];
};

f32_t apply_gamma(f32_t value, f32_t gamma) {
  return powf(value, 1.0f / gamma);
}

bool enumerate_monitors(std::vector<HMONITOR>& monitors) {
  auto MonitorEnumProc = [](HMONITOR hMonitor, HDC, LPRECT, LPARAM dwData) -> BOOL {
    std::vector<HMONITOR>* monitorList = reinterpret_cast<std::vector<HMONITOR>*>(dwData);
    monitorList->push_back(hMonitor);
    return TRUE;
    };

  return EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, reinterpret_cast<LPARAM>(&monitors));
}

bool get_monitor_info(HMONITOR hMonitor, MONITORINFOEX& monitorInfo) {
  monitorInfo.cbSize = sizeof(MONITORINFOEX);
  return ::GetMonitorInfo(hMonitor, &monitorInfo);
}

bool get_monitor_gamma(const std::string& device_name, gamma_ramp_t& gamma_ramp) {
  HDC hdc = CreateDC("DISPLAY", device_name.c_str(), NULL, NULL);
  if (!hdc) {
    fan::throw_error("failed to create device context for:" +  device_name);
    return false;
  }

  BOOL result = GetDeviceGammaRamp(hdc, &gamma_ramp);
  DeleteDC(hdc);
  return result != 0;
}

f32_t gamma_ramp_to_gamma(const gamma_ramp_t& ramp) {
  const int start = 64;   // start 25% of range
  const int end = 224;    // end ~88% of range
  f32_t gamma_sum = 0.0f;
  int valid_samples = 0;

  for (int i = start; i < end; i++) {
    f32_t value = static_cast<f32_t>(i) / 255.0f;

    f32_t r = static_cast<f32_t>(ramp.Red[i]) / 65535.0f;
    f32_t g = static_cast<f32_t>(ramp.Green[i]) / 65535.0f;
    f32_t b = static_cast<f32_t>(ramp.Blue[i]) / 65535.0f;
    f32_t v = (r + g + b) / 3.0f;

    if (v > 0.01f && value > 0.01f) {
      f32_t gamma_estimate = log(value) / log(v);

      if (isfinite(gamma_estimate) && !isnan(gamma_estimate) &&
        gamma_estimate > 0.5f && gamma_estimate < 5.0f) {
        gamma_sum += gamma_estimate;
        valid_samples++;
      }
    }
  }
  if (valid_samples < 10) {
    return 2.2f;
  }
  return gamma_sum / valid_samples;
}

bool set_monitor_gamma(const std::string& device_name, f32_t gamma) {
  HDC hdc = CreateDC("DISPLAY", device_name.c_str(), NULL, NULL);
  if (!hdc) {
    fan::throw_error("failed to create device context for:" +  device_name);
    return false;
  }

  gamma_ramp_t ramp;
  for (int i = 0; i < 256; i++) {
    f32_t value = static_cast<f32_t>(i) / 255.0f;
    f32_t correctedValue = apply_gamma(value, gamma);
    WORD corrected16Bit = static_cast<WORD>(correctedValue * 65535.0f + 0.5f);

    ramp.Red[i] = corrected16Bit;
    ramp.Green[i] = corrected16Bit;
    ramp.Blue[i] = corrected16Bit;
  }

  BOOL result = SetDeviceGammaRamp(hdc, &ramp);
  DeleteDC(hdc);
  return result != 0;
}

bool set_monitor_gamma_ramp(const std::string& device_name, const gamma_ramp_t& gamma_ramp) {
  HDC hdc = CreateDC("DISPLAY", device_name.c_str(), NULL, NULL);
  if (!hdc) {
    fan::throw_error("failed to create device context for:" +  device_name);
    return false;
  }

  BOOL result = SetDeviceGammaRamp(hdc, const_cast<gamma_ramp_t*>(&gamma_ramp));
  DeleteDC(hdc);
  return result != 0;
}

void set_monitors_gamma(std::vector<HMONITOR>& monitors, f32_t gamma) {
  for (size_t i = 0; i < monitors.size(); i++) {
    MONITORINFOEX monitorInfo;
    if (get_monitor_info(monitors[i], monitorInfo)) {
      set_monitor_gamma(monitorInfo.szDevice, gamma);
    }
  }
}

f32_t apply_contrast(f32_t value, f32_t contrast) {
  return (value - 0.5f) * contrast + 0.5f;
}

bool set_monitor_gamma_contrast(const std::string& device_name, f32_t gamma, f32_t contrast) {
  HDC hdc = CreateDC("DISPLAY", device_name.c_str(), NULL, NULL);
  if (!hdc) {
    fan::throw_error("failed to create device context for:" + device_name);
    return false;
  }

  gamma_ramp_t ramp;
  for (int i = 0; i < 256; i++) {
    f32_t value = static_cast<f32_t>(i) / 255.0f;

    f32_t contrasted_value = apply_contrast(value, contrast);
    contrasted_value = std::max(0.0f, std::min(1.0f, contrasted_value));

    f32_t corrected_value = apply_gamma(contrasted_value, gamma);
    WORD corrected_16bit = static_cast<WORD>(corrected_value * 65535.0f + 0.5f);

    ramp.Red[i] = corrected_16bit;
    ramp.Green[i] = corrected_16bit;
    ramp.Blue[i] = corrected_16bit;
  }

  BOOL result = SetDeviceGammaRamp(hdc, &ramp);
  DeleteDC(hdc);
  return result != 0;
}

void set_monitors_gamma_contrast(std::vector<HMONITOR>& monitors, f32_t gamma, f32_t contrast) {
  for (size_t i = 0; i < monitors.size(); i++) {
    MONITORINFOEX monitor_info;
    if (get_monitor_info(monitors[i], monitor_info)) {
      set_monitor_gamma_contrast(monitor_info.szDevice, gamma, contrast);
    }
  }
}

void render_set_gamma(std::vector<HMONITOR>& monitors, f32_t& gamma, f32_t& contrast) {
  ImGui::Begin("##gamma settings");

  if (ImGui::SliderFloat("gamma", &gamma, 0.3f, 2.8f, "%.2f")) {
    set_monitors_gamma(monitors, gamma);
  }
  if (ImGui::SliderFloat("contrast", &contrast, .5f, 2.f)) {
    set_monitors_gamma_contrast(monitors, gamma, contrast);
  }
  ImGui::End();
}

int main() {
  using namespace fan::graphics;

  /*glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
  glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);*/
  engine_t engine{ {
    .renderer = engine_t::renderer_t::opengl
  } };

  f32_t gamma = 1.f, contrast = 1.f;

  std::vector<HMONITOR> monitors;
  if (!enumerate_monitors(monitors)) {
    fan::throw_error("failed to enumerate monitors");
    return 1;
  }

  std::vector<gamma_ramp_t> original_gammas(monitors.size());
  for (size_t i = 0; i < monitors.size(); i++) {
    MONITORINFOEX monitorInfo;
    if (get_monitor_info(monitors[i], monitorInfo)) {
      if (!get_monitor_gamma(monitorInfo.szDevice, original_gammas[i])) {
        fan::print_warning("failed to get current gamma for:" + std::string(monitorInfo.szDevice));
      }
      else {
        gamma = gamma_ramp_to_gamma(original_gammas[i]);
      }
    }
  }

  {
    fan_window_loop{
      render_set_gamma(monitors, gamma, contrast);
    };
  }

  // restore
  for (size_t i = 0; i < monitors.size(); i++) {
    MONITORINFOEX monitorInfo;
    if (get_monitor_info(monitors[i], monitorInfo)) {
      set_monitor_gamma_ramp(monitorInfo.szDevice, original_gammas[i]);
    }
  }
}