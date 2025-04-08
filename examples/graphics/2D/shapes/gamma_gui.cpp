#include <fan/pch.h>

struct gamma_ramp_t {
  WORD Red[256];
  WORD Green[256];
  WORD Blue[256];
};

f32_t apply_gamma(f32_t value, f32_t gamma) {
  return powf(value, 1.0f / gamma);
}

struct monitor_t {
  HMONITOR monitor;
  std::string name;
  f32_t gamma = 1;
  f32_t contrast = 1;
  bool enabled = 1;
};

bool enumerate_monitors(std::vector<monitor_t>& monitors) {
  auto MonitorEnumProc = [](HMONITOR hMonitor, HDC, LPRECT, LPARAM dwData) -> BOOL {
    std::vector<monitor_t>* monitorList = reinterpret_cast<std::vector<monitor_t>*>(dwData);
    MONITORINFOEX monitorInfo;
    monitorInfo.cbSize = sizeof(MONITORINFOEX);
    if (GetMonitorInfo(hMonitor, &monitorInfo)) {
      monitorList->push_back(monitor_t{.monitor=hMonitor, .name=monitorInfo.szDevice});
    }
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

bool internal_set_monitor_gamma(const std::string& device_name, f32_t gamma) {
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

void set_monitor_gamma(const monitor_t& monitor, f32_t gamma) {
  MONITORINFOEX monitorInfo;
  if (get_monitor_info(monitor.monitor, monitorInfo)) {
    internal_set_monitor_gamma(monitorInfo.szDevice, gamma);
  }
}

void set_monitors_gamma(const std::vector<monitor_t>& monitors, f32_t gamma) {
  for (size_t i = 0; i < monitors.size(); i++) {
    set_monitor_gamma(monitors[i], gamma);
  }
}

f32_t apply_contrast(f32_t value, f32_t contrast) {
  return (value - 0.5f) * contrast + 0.5f;
}

bool global_gamma = true;

bool internal_set_monitor_gamma_contrast(const std::string& device_name, f32_t gamma, f32_t contrast) {
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

void set_monitor_gamma_contrast(const monitor_t& monitor) {
  MONITORINFOEX monitor_info;
  if (get_monitor_info(monitor.monitor, monitor_info)) {
    internal_set_monitor_gamma_contrast(monitor_info.szDevice, monitor.gamma, monitor.contrast);
  }
}

void set_monitors_gamma_contrast(const std::vector<monitor_t>& monitors, f32_t gamma, f32_t contrast) {
  for (size_t i = 0; i < monitors.size(); i++) {
    MONITORINFOEX monitor_info;
    if (get_monitor_info(monitors[i].monitor, monitor_info)) {
      internal_set_monitor_gamma_contrast(monitor_info.szDevice, gamma, contrast);
    }
  }
}

bool change_gamma_when_going_in_game = true;

void render_set_gamma(std::vector<monitor_t>& monitors, f32_t& gamma, f32_t& contrast) {
  ImGui::Begin("##gamma settings");

  ImGui::Checkbox("Change gamma when going in game", &change_gamma_when_going_in_game);

  if (ImGui::Checkbox("Global gamma", &global_gamma)) {
    set_monitors_gamma_contrast(monitors, gamma, contrast);
  }

  if (global_gamma == false) {
    for (auto [i, monitor] : fan::enumerate(monitors)) {
      if (ImGui::Checkbox(("Monitor #" + std::to_string(i)).c_str(), &monitor.enabled)) {
        set_monitor_gamma_contrast(monitors[i]);
      }
      if (monitor.enabled == false) {
        continue;
      }

      ImGui::PushID(i);
      if (ImGui::SliderFloat("gamma", &monitors[i].gamma, 0.3f, 2.8f, "%.2f")) {
        set_monitor_gamma_contrast(monitors[i]);
      }
      if (ImGui::SliderFloat("contrast", &monitors[i].contrast, .5f, 2.f)) {
        set_monitor_gamma_contrast(monitors[i]);
      }
      ImGui::PopID();
    }
  }
  else {
    if (ImGui::SliderFloat("gamma", &gamma, 0.3f, 2.8f, "%.2f")) {
      set_monitors_gamma_contrast(monitors, gamma, contrast);
    }
    if (ImGui::SliderFloat("contrast", &contrast, .5f, 2.f)) {
      set_monitors_gamma_contrast(monitors, gamma, contrast);
    }
  }

  ImGui::End();
}

void restore_original_gammas(const std::vector<monitor_t>& monitors, const std::vector<gamma_ramp_t>& original_gammas) {
  for (size_t i = 0; i < monitors.size(); i++) {
    MONITORINFOEX monitorInfo;
    if (get_monitor_info(monitors[i].monitor, monitorInfo)) {
      set_monitor_gamma_ramp(monitorInfo.szDevice, original_gammas[i]);
    }
  }
}

int main() {
  using namespace fan::graphics;
  engine_t engine{ {
    .renderer = engine_t::renderer_t::opengl
  } };

  std::string file_name = "gamma_config.json";

  f32_t gamma = 1.f, contrast = 1.f;
  std::vector<monitor_t> valid_monitors;
  if (!enumerate_monitors(valid_monitors)) {
    fan::throw_error("failed to enumerate monitors");
    return 1;
  }
  std::vector<monitor_t> monitors;

  std::vector<gamma_ramp_t> original_gammas(valid_monitors.size());
  for (size_t i = 0; i < valid_monitors.size(); i++) {
    MONITORINFOEX monitorInfo;
    if (get_monitor_info(valid_monitors[i].monitor, monitorInfo)) {
      if (!get_monitor_gamma(monitorInfo.szDevice, original_gammas[i])) {
        fan::printcl_warn("failed to get current gamma for:" + std::string(monitorInfo.szDevice));
      }
    }
  }

  fan::json json_config;
  auto load_config = [&] {
    if (fan::io::file::exists(file_name)) {
      std::string data;
      fan::io::file::read(file_name, &data);
      json_config = fan::json::parse(data);
      if (json_config.contains("global_gamma")) {
        global_gamma = json_config["global_gamma"];
        if (json_config.contains("gamma")) {
          gamma = json_config["gamma"];
        }
        if (json_config.contains("contrast")) {
          contrast = json_config["contrast"];
        }
        if (json_config.contains("change_gamma_when_going_in_game")) {
          change_gamma_when_going_in_game = json_config["change_gamma_when_going_in_game"];
        }
        set_monitors_gamma_contrast(valid_monitors, gamma, contrast);
      }
      if (json_config.contains("monitors") == false) {
        fan::printcl_warn("json key \"monitors\" not found");
        return -1;
      }
      fan::json json_monitors = json_config["monitors"];
      for (auto& [index, monitor_data] : json_monitors.items()) {
        std::string monitor_name = monitor_data["name"];
        auto found = std::find_if(valid_monitors.begin(), valid_monitors.end(), [&monitor_name](const monitor_t& a) {
          return a.name == monitor_name;
          });
        if (found == valid_monitors.end()) {
          fan::printcl_warn("failed to find monitor:" + monitor_name + ", skipping...");
          continue;
        }
        monitor_t monitor;
        monitor.monitor = found->monitor;
        monitor.name = monitor_name;
        monitor.enabled = monitor_data["enabled"];
        monitor.gamma = monitor_data["gamma"];
        monitor.contrast = monitor_data["contrast"];
        monitors.push_back(monitor);
        if (!global_gamma) {
          set_monitor_gamma_contrast(monitor);
        }
      }
    }
    else {
      monitors = valid_monitors;
      if (original_gammas.size()) {
        gamma = gamma_ramp_to_gamma(original_gammas.front());
      }
    }
  };
  load_config();
  static auto save_config = [&]{
    json_config["global_gamma"] = global_gamma;
    json_config["gamma"] = gamma;
    json_config["contrast"] = contrast;
    json_config["change_gamma_when_going_in_game"] = change_gamma_when_going_in_game;
    fan::json json_monitors = fan::json::array();
    for (const monitor_t& monitor : monitors) {
      fan::json json_monitor;
      json_monitor["name"] = monitor.name;
      json_monitor["enabled"] = monitor.enabled;
      json_monitor["gamma"] = monitor.gamma;
      json_monitor["contrast"] = monitor.contrast;
      json_monitors.push_back(json_monitor);
    }
    json_config["monitors"] = json_monitors;
    fan::io::file::write(file_name, json_config.dump(2), std::ios_base::binary);
  };

  struct save_config_t {
    ~save_config_t() {
      save_config();
    }
  }save;

  {
    std::string window_title;
    HWND top_window;
    std::string previous_title;
    int state = -1;
    fan_window_loop{
      render_set_gamma(monitors, gamma, contrast);
      if (change_gamma_when_going_in_game) {
        top_window = GetForegroundWindow();
        window_title.resize(256);
        int len = GetWindowText(top_window, window_title.data(), window_title.size());
        if (len > 0) {
          window_title.resize(len);
        }
        else {
          window_title.clear();
        }
        if (previous_title != window_title) {
          if (window_title == "EscapeFromTarkov" && state != 0) {
            if (global_gamma) {
              set_monitors_gamma_contrast(monitors, gamma, contrast);
            }
            else {
              for (const monitor_t& monitor : monitors) {
                set_monitor_gamma_contrast(monitor);
              }
            }
            state = 0;
          }
          else if (state != 1) {
            restore_original_gammas(monitors, original_gammas);
            state = 1;
          }
        }
        previous_title = window_title;
      }
    };
  }
  restore_original_gammas(monitors, original_gammas);
}