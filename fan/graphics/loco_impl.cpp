module;

#if defined (FAN_WINDOW)

// loco framebuffer is recommended, you cant see sprites without it, 
// since light uses framebuffer _t01. you could use unlit_sprite, if required
#include <coroutine>

// TODO REMOVE
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <fan/graphics/shape_macros.h>

#include <fan/utility.h>

#if defined(fan_platform_windows)
  #include <Windows.h>
#endif

#undef min
#undef max



#endif

module fan.graphics.loco;

#if defined (FAN_WINDOW)

import std;

import fan.window.input;

import fan.event.types;

import fan.camera;
import fan.memory;
import fan.random;
import fan.print;

import fan.event.uv_raw;

import fan.graphics.common_types;

#if defined(FAN_GUI)
  import fan.graphics.gui.types;
  import fan.graphics.gui.text_logger;
  import fan.graphics.gui.settings_menu;
#endif


import fan.io.file;

using namespace fan::graphics;

#if defined(FAN_GUI)
namespace gui = fan::graphics::gui;

::gui::settings_menu_t* get_smenu(loco_t* loco) {
  return static_cast<::gui::settings_menu_t*>(loco->gui.settings_menu);
}

namespace fan {
  namespace graphics {
    namespace gui {
      void render_allocations_plot();
      void process_frame();
    }
  }
}
#endif

global_loco_t& gloco() {
  static global_loco_t loco;
  return loco;
}

fan::graphics::engine_init_t::init_callback_t& fan::graphics::get_engine_init_cbs() {
  static fan::graphics::engine_init_t::init_callback_t engine_init_cbs;
  return engine_init_cbs;
}

template<typename list_t, typename fn_t>
static void for_each_list(list_t& list, fn_t&& fn) {
  typename list_t::nrtra_t nrtra;
  typename list_t::nr_t nr;
  nrtra.Open(&list, &nr);
  while (nrtra.Loop(&list, &nr)) {
    fn(list, nr);
  }
  nrtra.Close(&list);
}

struct loco_t::vulkan_t {
  loco_t* loco_ptr = nullptr;

  #include <fan/graphics/vulkan/engine_functions.h>

  fan::vulkan::context_t::pipeline_t post_process;
  VkResult image_error{};
  fan::window_t::resize_handle_t window_resize_handle;
};

namespace fan::graphics {
  bool async_image_t::ready() const {
    return result != nullptr && result->state == fan::image::async_result_t::state_e::ready;
  }

  bool async_image_t::failed() const {
    return result != nullptr && result->state == fan::image::async_result_t::state_e::failed;
  }

  
  void fan::graphics::time_monitor_t::update(f32_t v) {
    if (paused || v <= 0.0f) return;

    buffer.push_back(v);
    sum += v;

    int idx = static_cast<int>(buffer.size()) - 1;

    while (!min_q.empty() && buffer[min_q.back()] >= v) {
      min_q.pop_back();
    }
    min_q.push_back(idx);

    while (!max_q.empty() && buffer[max_q.back()] <= v) {
      max_q.pop_back();
    }
    max_q.push_back(idx);
  }

  void fan::graphics::time_monitor_t::reset() {
    buffer.clear();
    sum = 0.0f;
    min_q.clear();
    max_q.clear();
  }

  fan::graphics::time_monitor_t::stats_t fan::graphics::time_monitor_t::stats() const {
    if (buffer.empty()) return {0, 0, 0};

    return {
      sum / buffer.size(),
      buffer[min_q.front()],
      buffer[max_q.front()]
    };
  }

  #if defined(FAN_GUI)
  void fan::graphics::time_monitor_t::plot(loco_t* loco, std::string_view label) {
    using namespace fan::graphics;
    if (buffer.empty()) return;

    int plot_count = std::min(loco->gui.time_plot_scroll.view_size, static_cast<int>(buffer.size()));
    static std::vector<f32_t> plot_data;
    plot_data.resize(plot_count);

    if (!paused) {
      int max_start = std::max(0, static_cast<int>(buffer.size()) - loco->gui.time_plot_scroll.view_size);
      loco->gui.time_plot_scroll.scroll_offset = max_start;
    }

    int max_start = std::max(0, static_cast<int>(buffer.size()) - loco->gui.time_plot_scroll.view_size);
    int start = std::min(loco->gui.time_plot_scroll.scroll_offset, max_start);

    for (int i = 0; i < plot_count; ++i) {
      plot_data[i] = buffer[start + i] * 1e3f; // ms
    }

    ::gui::plot::plot_line(label, plot_data.data(), plot_count);
  }
  #endif

  std::uint32_t get_draw_mode(std::uint8_t internal_draw_mode) {
    return fan::vulkan::core::get_draw_mode(internal_draw_mode);
  }
}

fan::window_t& loco_t::get_window() {
  return window;
}

fan::graphics::shader_nr_t loco_t::shader_create() {
  return context_functions.shader_create(&context);
}

fan::graphics::context_shader_t loco_t::shader_get(fan::graphics::shader_nr_t nr) {
  fan::graphics::context_shader_t obj {};
  obj.vk = fan::graphics::get_vk_context().shaders.shader_get(nr);
  return obj;
}

void loco_t::shader_erase(fan::graphics::shader_nr_t nr) {
  context_functions.shader_erase(&context, nr);
}

void loco_t::shader_use(fan::graphics::shader_nr_t nr) {
  context_functions.shader_use(&context, nr);
}

void loco_t::shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& vertex_code) {
  context_functions.shader_set_vertex(&context, nr, file_path, vertex_code);
}

void loco_t::shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string_view file_path, const std::string& fragment_code) {
  context_functions.shader_set_fragment(&context, nr, file_path, fragment_code);
}

void loco_t::shader_set_compute(
  fan::graphics::shader_nr_t nr,
  const std::string_view file_path,
  const std::string& compute_code
) {
  context_functions.shader_set_compute(
    &context,
    nr,
    file_path,
    compute_code
  );
}

bool loco_t::shader_compile(fan::graphics::shader_nr_t nr) {
  return context_functions.shader_compile(&context, nr);
}

#if defined(FAN_2D)

void loco_t::shader_set_camera(fan::graphics::shader_nr_t nr, camera_t camera_nr) {
  fan::graphics::get_vk_context().shaders.shader_set_camera(nr, camera_nr); 
}

fan::graphics::shader_nr_t loco_t::shader_get_nr(std::uint16_t shape_type) {
  return fan::graphics::g_shapes->shaper.GetShader(shape_type);
}

fan::graphics::shader_list_t::nd_t& loco_t::shader_get_data(std::uint16_t shape_type) {
  return loco_t::shader_list[shader_get_nr(shape_type)];
}

fan::graphics::shader_list_t::nd_t& loco_t::shader_get_data(fan::graphics::shader_t shader) {
  return loco_t::shader_list[shader];
}

void loco_t::shader_set_paths(fan::graphics::shader_t shader, std::string_view vertex, std::string_view fragment) {
  auto& sdata = shader_get_data(shader);
  sdata.path_vertex = vertex;
  sdata.path_fragment = fragment;
}

void loco_t::shader_recompile_all() {
  for_each_list(shader_list, [&](auto& list, auto nr) {
    auto& sd = list[nr];
    if ((sd.svertex.empty() || sd.sfragment.empty()) && sd.scompute.empty()) return;

    auto read = [](const auto& path, const auto& fallback) {
      std::string src = fan::graphics::read_shader(path);
      if (src.empty() && !fallback.empty() && !std::string_view(path).empty()) {
        std::string spath(path);
        fan::print_warning("failed to read shader file path:" +
          (spath.empty() ? "FILE PATH NOT FOUND" : spath) + ", assigning old compiled shader.");
      }
      return src.empty() ? fallback : src;
    };

    if (!sd.scompute.empty()) {
      auto sc = read(sd.path_compute, sd.scompute);
      if (sc == sd.scompute) return;
      shader_set_compute(nr, sd.path_compute, sc);
      if (!shader_compile(nr)) {
        fan::print_warning("failed to recompile compute shader. compute shader:" + std::string(sd.path_compute));
      }
      return;
    }

    auto sv = read(sd.path_vertex, sd.svertex);
    auto sf = read(sd.path_fragment, sd.sfragment);
    if (sv == sd.svertex && sf == sd.sfragment) return;

    if (sv != sd.svertex)   shader_set_vertex(nr, sd.path_vertex, sv);
    if (sf != sd.sfragment) shader_set_fragment(nr, sd.path_fragment, sf);
    if (!shader_compile(nr))
      fan::print_warning("failed to recompile shader. vertex shader:" + std::string(sd.path_vertex) + ", fragment shader:" + std::string(sd.path_fragment));
  });
}

f32_t* loco_t::get_bloom_filter_radius_ptr() { return &vk->bloom_filter_radius; }
f32_t* loco_t::get_bloom_threshold_ptr()     { return &vk->bloom_threshold; }
f32_t* loco_t::get_bloom_knee_ptr()          { return &vk->bloom_knee; }
f32_t* loco_t::get_bloom_smooth_rate_ptr()   { return &vk->bloom_smooth_rate; }
f32_t* loco_t::get_bloom_luma_scale_ptr()    { return &vk->bloom_luma_scale; }
f32_t* loco_t::get_bloom_adaptation_blend_ptr() { return &vk->bloom_adaptation_blend; }
fan::vec3* loco_t::get_bloom_tint_ptr()      { return &vk->bloom_tint; }
f32_t* loco_t::get_bloom_strength_ptr()      { return &vk->bloom_strength; }
f32_t* loco_t::get_gamma_ptr()               { return &vk->gamma; }
f32_t* loco_t::get_exposure_ptr()            { return &vk->exposure; }
f32_t* loco_t::get_contrast_ptr()            { return &vk->contrast; }

void loco_t::set_settings(const post_process_settings_t& settings) {
  if (settings.clear_color) set_clear_color(*settings.clear_color);
  if (settings.ambient_color) get_lighting().set_target(*settings.ambient_color, 0.0f);
  if (settings.mode) open_props.post_process_mode = *settings.mode;
  if (settings.bloom_strength) *get_bloom_strength_ptr() = *settings.bloom_strength;
  if (settings.bloom_threshold) *get_bloom_threshold_ptr() = *settings.bloom_threshold;
  if (settings.bloom_knee) *get_bloom_knee_ptr() = *settings.bloom_knee;
  if (settings.bloom_smooth_rate) *get_bloom_smooth_rate_ptr() = *settings.bloom_smooth_rate;
  if (settings.bloom_luma_scale) *get_bloom_luma_scale_ptr() = *settings.bloom_luma_scale;
  if (settings.bloom_adaptation_blend) *get_bloom_adaptation_blend_ptr() = *settings.bloom_adaptation_blend;
  if (settings.bloom_tint) *get_bloom_tint_ptr() = *settings.bloom_tint;
  if (settings.bloom_filter_radius) *get_bloom_filter_radius_ptr() = *settings.bloom_filter_radius;
  if (settings.blur_amount) open_props.blur_amount = *settings.blur_amount;
  if (settings.blur_filter_radius) open_props.blur_filter_radius = *settings.blur_filter_radius;
  if (settings.blur_focus_enabled) open_props.blur_focus_enabled = *settings.blur_focus_enabled;
  if (settings.gamma) *get_gamma_ptr() = *settings.gamma;
  if (settings.exposure) *get_exposure_ptr() = *settings.exposure;
  if (settings.contrast) *get_contrast_ptr() = *settings.contrast;

#if defined(FAN_GUI)
  auto* sm = get_smenu(this);
  if (sm) {
    auto& pp = sm->config.post_processing;
    if (settings.bloom_strength) pp.bloom_strength = *settings.bloom_strength;
    if (settings.bloom_threshold) pp.bloom_threshold = *settings.bloom_threshold;
    if (settings.bloom_knee) pp.bloom_knee = *settings.bloom_knee;
    if (settings.bloom_smooth_rate) pp.bloom_smooth_rate = *settings.bloom_smooth_rate;
    if (settings.bloom_luma_scale) pp.bloom_luma_scale = *settings.bloom_luma_scale;
    if (settings.bloom_adaptation_blend) pp.bloom_adaptation_blend = *settings.bloom_adaptation_blend;
    if (settings.bloom_tint) pp.bloom_tint = *settings.bloom_tint;
    if (settings.bloom_filter_radius) pp.bloom_filter_radius = *settings.bloom_filter_radius;
    if (settings.gamma) pp.gamma = *settings.gamma;
    if (settings.exposure) pp.exposure = *settings.exposure;
    if (settings.contrast) pp.contrast = *settings.contrast;
  }
#endif
}
#endif


void loco_t::shadow_add_caster(fan::graphics::shape_t* shape, f32_t alpha_threshold) {
}

void loco_t::shadow_remove_caster(fan::graphics::shape_t* shape) {
}

void loco_t::shadow_clear_casters() {
}

void loco_t::shadow_add_light(
  fan::vec2 position,
  f32_t radius,
  fan::color color,
  f32_t softness,
  f32_t falloff_power,
  f32_t angle,
  f32_t cone_inner,
  f32_t cone_outer
) {
}

void loco_t::shadow_set_light_angle(std::size_t index, f32_t angle) {
}

void loco_t::shadow_set_light_cone(std::size_t index, f32_t cone_inner, f32_t cone_outer) {
}
void loco_t::shadow_set_light_position(std::size_t index, fan::vec2 position) {
}

void loco_t::shadow_clear_lights() {
}

void loco_t::shadow_set_darkness(f32_t darkness) {
}

std::size_t loco_t::shadow_light_count() {
  return 0;
}

std::vector<std::uint8_t> loco_t::image_get_pixel_data(
  fan::graphics::image_t nr,
  int image_format,
  fan::vec2 uvp,
  fan::vec2 uvs
) {
  return fan::graphics::get_vk_context().image_get_pixel_data(
    nr,
    image_format,
    uvp,
    uvs
  );
}

fan::graphics::image_t loco_t::image_create() {
  return context_functions.image_create(&context);
}

fan::graphics::context_image_t loco_t::image_get(fan::graphics::image_t nr) {
  fan::graphics::context_image_t obj {};
  obj.vk = fan::graphics::get_vk_context().image_get(nr);
  return obj;
}

std::uint64_t loco_t::image_get_handle(fan::graphics::image_t nr) {
  return context_functions.image_get_handle(&context, nr);
}

fan::graphics::image_data_t& loco_t::image_get_data(fan::graphics::image_t nr) {
  return image_list[nr];
}

void loco_t::image_erase(fan::graphics::image_t nr) {
  context_functions.image_erase(&context, nr);
}

void loco_t::image_bind(fan::graphics::image_t nr) {
  context_functions.image_bind(&context, nr);
}

void loco_t::image_unbind(fan::graphics::image_t nr) {
  context_functions.image_unbind(&context, nr);
}

fan::graphics::image_load_properties_t& loco_t::image_get_settings(fan::graphics::image_t nr) {
  return context_functions.image_get_settings(&context, nr);
}

void loco_t::image_set_settings(fan::graphics::image_t nr, const fan::graphics::image_load_properties_t& settings) {
  context_functions.image_set_settings(&context, nr, settings);
}

fan::graphics::image_t loco_t::image_load(const fan::image::info_t& image_info) {
  return context_functions.image_load_info(&context, image_info);
}

fan::graphics::image_t loco_t::image_load(const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
  return context_functions.image_load_info_props(&context, image_info, p);
}

fan::graphics::image_t loco_t::image_load(const std::string& path, const std::source_location& callers_path) {
  return context_functions.image_load_path(&context, path, callers_path);
}

fan::graphics::image_t loco_t::image_load(const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path) {
  return context_functions.image_load_path_props(&context, path, p, callers_path);
}

fan::graphics::image_t loco_t::request_image_load_async(const std::string& path, const fan::graphics::image_load_properties_t& p, std::function<void(const fan::graphics::decoded_image_payload_t&)> on_gpu_uploaded) {
  return context_functions.request_image_load_async(&context, path, p, on_gpu_uploaded);
}

void loco_t::process_async_image_uploads() {
  context_functions.process_async_image_uploads(&context);
}

void loco_t::flush_startup_async_images() {
  while (!context.vk.pending_image_uploads.empty() || fan::image::has_pending_async_tasks()) {
    process_async_image_uploads();
    std::this_thread::yield();
  }
}

fan::graphics::image_t loco_t::image_load(fan::color* colors, const fan::vec2ui& size) {
  return context_functions.image_load_colors(&context, colors, size);
}

fan::graphics::image_t loco_t::image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
  return context_functions.image_load_colors_props(&context, colors, size, p);
}

void loco_t::image_unload(fan::graphics::image_t nr) {
  context_functions.image_unload(&context, nr);
}

bool loco_t::is_image_valid(fan::graphics::image_t nr) {
  return nr != default_texture && nr.iic() == false;
}

fan::graphics::image_t loco_t::create_missing_texture() {
  return context_functions.create_missing_texture(&context);
}

fan::graphics::image_t loco_t::create_transparent_texture() {
  return context_functions.create_transparent_texture(&context);
}

void loco_t::image_reload(fan::graphics::image_t nr, const fan::image::info_t& image_info) {
  context_functions.image_reload_image_info(&context, nr, image_info);
}

void loco_t::image_reload(fan::graphics::image_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
  context_functions.image_reload_image_info_props(&context, nr, image_info, p);
}

void loco_t::image_reload(fan::graphics::image_t nr, const std::string& path, const std::source_location& callers_path) {
  context_functions.image_reload_path(&context, nr, path, callers_path);
}

void loco_t::image_reload(fan::graphics::image_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path) {
  context_functions.image_reload_path_props(&context, nr, path, p, callers_path);
}

fan::graphics::image_t loco_t::image_create(const fan::color& color) {
  return context_functions.image_create_color(&context, color);
}

fan::graphics::image_t loco_t::image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p) {
  return context_functions.image_create_color_props(&context, color, p);
}

fan::graphics::camera_nr_t loco_t::camera_create() {
  return context_functions.camera_create(&context);
}

fan::graphics::context_camera_t& loco_t::camera_get(fan::graphics::camera_nr_t nr) {
  return context_functions.camera_get(&context, nr);
}

void loco_t::camera_erase(fan::graphics::camera_nr_t nr) {
  context_functions.camera_erase(&context, nr);
}

fan::graphics::camera_nr_t loco_t::camera_create(const fan::vec2& x, const fan::vec2& y) {
  return context_functions.camera_create_params(&context, x, y);
}

fan::vec3 loco_t::camera_get_position(fan::graphics::camera_nr_t nr) {
  return context_functions.camera_get_position(&context, nr);
}

void loco_t::camera_set_position(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
  context_functions.camera_set_position(&context, nr, cp);
}

void loco_t::camera_set_position(const fan::vec3& cp) {
  camera_set_position(orthographic_render_view, cp);
}

fan::vec3 loco_t::camera_get_center(fan::graphics::camera_nr_t nr) {
  return context_functions.camera_get_center(&context, nr);
}

void loco_t::camera_set_center(fan::graphics::camera_nr_t nr, const fan::vec3& cp) {
  context_functions.camera_set_center(&context, nr, cp);
}

void loco_t::camera_set_center(const fan::vec3& cp) {
  camera_set_center(orthographic_render_view, cp);
}

fan::vec2 loco_t::camera_get_size(fan::graphics::camera_nr_t nr) {
  return context_functions.camera_get_size(&context, nr);
}

f32_t loco_t::camera_get_zoom(fan::graphics::camera_nr_t nr) {
  return context_functions.camera_get_zoom(&context, nr);
}

void loco_t::camera_set_zoom(fan::graphics::camera_nr_t nr, f32_t new_zoom) {
  context_functions.camera_set_zoom(&context, nr, new_zoom);
}

void loco_t::camera_set_ortho(fan::graphics::camera_nr_t nr, fan::vec2 x, fan::vec2 y) {
  context_functions.camera_set_ortho(&context, nr, x, y);
}

void loco_t::camera_set_perspective(fan::graphics::camera_nr_t nr, f32_t fov, const fan::vec2& window_size) {
  context_functions.camera_set_perspective(&context, nr, fov, window_size);
}

void loco_t::camera_rotate(fan::graphics::camera_nr_t nr, const fan::vec2& offset) {
  context_functions.camera_rotate(&context, nr, offset);
}

void loco_t::camera_rotate(const fan::vec2& offset) {
  camera_rotate(perspective_render_view, offset);
}

void loco_t::camera_follow(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed) {
  auto& c = context_functions.camera_get(&context, nr);
  fan::vec2 offset = fan::vec2(c.coordinates.left + c.coordinates.right, c.coordinates.top + c.coordinates.bottom) / (2.f * c.zoom);
  
  fan::vec2 src_center = camera_get_position(nr) + offset;
  
  f32_t t = 1.0f - std::exp(-move_speed * get_delta_time());
  
  camera_set_center(
    nr, 
    fan::vec3(move_speed == 0 ? target : src_center + (target - src_center) * t, 0.f)
  );
}

void loco_t::camera_follow(const fan::vec2& target, f32_t move_speed) {
  camera_follow(orthographic_render_view, target, move_speed);
}

void loco_t::camera_follow(fan::graphics::camera_nr_t nr, const fan::graphics::shapes::shape_t& shape, f32_t move_speed) {
  camera_follow(nr, shape.get_position(), move_speed);
}

void loco_t::camera_follow(const fan::graphics::shapes::shape_t& shape, f32_t move_speed) {
  camera_follow(shape.get_position(), move_speed);
}

loco_t::update_callback_handle_t loco_t::camera_set_target(
  fan::graphics::camera_nr_t nr,
 const fan::graphics::shapes::shape_t& shape, 
  f32_t move_speed) 
{
  return add_update_callback([&shape, move_speed, nr](void* loco) {
    ((loco_t*)loco)->camera_follow(nr, shape, move_speed);
  });
}

loco_t::update_callback_handle_t loco_t::camera_set_target(const fan::graphics::shapes::shape_t& shape, f32_t move_speed) {
  return camera_set_target(orthographic_render_view, shape, move_speed);
}

fan::graphics::viewport_nr_t loco_t::viewport_create() {
  return context_functions.viewport_create(&context);
}

fan::graphics::viewport_nr_t loco_t::viewport_create(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
  return context_functions.viewport_create_params(&context, viewport_position, viewport_size, window.get_size());
}

fan::graphics::context_viewport_t& loco_t::viewport_get(fan::graphics::viewport_nr_t nr) {
  return context_functions.viewport_get(&context, nr);
}

void loco_t::viewport_erase(fan::graphics::viewport_nr_t nr) {
  context_functions.viewport_erase(&context, nr);
}

fan::vec2 loco_t::viewport_get_position(fan::graphics::viewport_nr_t nr) {
  return context_functions.viewport_get_position(&context, nr);
}

fan::vec2 loco_t::viewport_get_size(fan::graphics::viewport_nr_t nr) {
  return context_functions.viewport_get_size(&context, nr);
}

void loco_t::viewport_set(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
  context_functions.viewport_set(&context, viewport_position, viewport_size, window.get_size());
}

void loco_t::viewport_set(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
  context_functions.viewport_set_nr(&context, nr, viewport_position, viewport_size, window.get_size());
}

void loco_t::viewport_set_size(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_size) {
  context_functions.viewport_set_nr(&context, nr, viewport_get_position(nr), viewport_size, window.get_size());
}

void loco_t::viewport_set_position(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position) {
  context_functions.viewport_set_nr(&context, nr, viewport_position, viewport_get_size(nr), window.get_size());
}

void loco_t::viewport_zero(fan::graphics::viewport_nr_t nr) {
  context_functions.viewport_zero(&context, nr);
}

bool loco_t::inside(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
  return context_functions.viewport_inside(&context, nr, position);
}

bool loco_t::inside_wir(fan::graphics::viewport_nr_t nr, const fan::vec2& position) {
  return context_functions.viewport_inside_wir(&context, nr, position);
}

bool loco_t::inside(const fan::graphics::render_view_t& render_view, const fan::vec2& position) {
  fan::vec2 tp = translate_position(position, render_view.viewport, render_view.camera);

  auto c = camera_get(render_view.camera);
  f32_t l = c.coordinates.left;
  f32_t r = c.coordinates.right;
  f32_t t = c.coordinates.top;
  f32_t b = c.coordinates.bottom;

  return tp.x >= l && tp.x <= r && tp.y >= t && tp.y <= b;
}

bool loco_t::is_mouse_inside(const fan::graphics::render_view_t& render_view) {
  return inside(render_view, get_mouse_position());
}

void loco_t::use() {
  gloco() = this;
  window.make_context_current();
}

void loco_t::camera_move(f32_t movement_speed, f32_t friction) {
  camera_move(camera_get(perspective_render_view), movement_speed, friction);
}

void loco_t::camera_move(fan::graphics::context_camera_t& camera, f32_t movement_speed, f32_t friction) {
  auto dt = get_delta_time();
  camera.velocity /= friction * dt + 1;
  static constexpr auto minimum_velocity = 0.001;
  if (camera.velocity.x < minimum_velocity && camera.velocity.x > -minimum_velocity) {
    camera.velocity.x = 0;
  }
  if (camera.velocity.y < minimum_velocity && camera.velocity.y > -minimum_velocity) {
    camera.velocity.y = 0;
  }
  if (camera.velocity.z < minimum_velocity && camera.velocity.z > -minimum_velocity) {
    camera.velocity.z = 0;
  }

  f64_t msd = movement_speed * dt;
  if (is_key_down(fan::input::key_w)) { camera.velocity += camera.front * msd; }
  if (is_key_down(fan::input::key_s)) { camera.velocity -= camera.front * msd; }
  if (is_key_down(fan::input::key_a)) { camera.velocity -= camera.right * msd; }
  if (is_key_down(fan::input::key_d)) { camera.velocity += camera.right * msd; }
  if (is_key_down(fan::input::key_space)) { camera.velocity.y += msd; }
  if (is_key_down(fan::input::key_left_shift)) { camera.velocity.y -= msd; }

  f64_t rotate = camera.sensitivity * get_delta_time();
  if (is_key_down(fan::input::key_left)) { camera.set_yaw(camera.get_yaw() - rotate); }
  if (is_key_down(fan::input::key_right)) { camera.set_yaw(camera.get_yaw() + rotate); }
  if (is_key_down(fan::input::key_up)) { camera.set_pitch(camera.get_pitch() + rotate); }
  if (is_key_down(fan::input::key_down)) { camera.set_pitch(camera.get_pitch() - rotate); }

  camera.position += camera.velocity * get_delta_time();
  camera.update_view();
  camera.view = camera.get_view_matrix();
}

#if defined(FAN_2D)

void loco_t::add_shape_to_immediate_draw(fan::graphics::shapes::shape_t&& s) {
  immediate_render_list.emplace_back(std::move(s));
}

std::uint32_t loco_t::add_shape_to_static_draw(fan::graphics::shapes::shape_t&& s) {
  std::uint32_t ret = s.NRI;
  static_render_list[ret] = std::move(s);
  return ret;
}

void loco_t::remove_static_shape_draw(const fan::graphics::shapes::shape_t& s) {
  static_render_list.erase(s.NRI);
}
#endif

static void add_simple_command(
  fan::console_t& console,
  const std::string& name,
  const std::string& desc,
  int arg_count,
  std::function<void(loco_t*, const std::string&)> fn)
{
  console.commands.add(name, [fn, arg_count](fan::console_t* self, const auto& args) {
    auto* loco = OFFSETLESS(self, loco_t, gui.console);
    if ((int)args.size() != arg_count) {
      loco->gui.console.commands.print_invalid_arg_count();
      return;
    }
    fn(loco, args.empty() ? "" : args[0]);
  }).description = desc;
}


void loco_t::generate_commands(loco_t* loco) {
#if defined(FAN_GUI)
  loco->gui.console.open();

  add_simple_command(loco->gui.console, "set_vsync", "sets vsync", 1,
    [](loco_t* l, const std::string& v) { l->set_vsync(std::stoi(v)); });
  add_simple_command(loco->gui.console, "set_target_fps", "sets target fps", 1,
    [](loco_t* l, const std::string& v) { l->set_target_fps(std::stoi(v)); });
  add_simple_command(loco->gui.console, "gui.show_fps", "toggles fps", 1,
    [](loco_t* l, const std::string& v) { l->gui.show_fps = std::stoi(v); });
  add_simple_command(loco->gui.console, "debug_memory", "opens memory debug", 1,
    [](loco_t* l, const std::string& v) { l->gui.render_debug_memory = std::stoi(v); });
  add_simple_command(loco->gui.console, "set_clear_color", "sets clear color - example {1,0,0,1}", 1,
    [](loco_t* l, const std::string& v) { l->renderer_state.clear_color = fan::color::parse(v); });
  add_simple_command(loco->gui.console, "set_lighting_ambient", "sets lighting ambient color", 1,
    [](loco_t* l, const std::string& v) { l->renderer_state.lighting.set_target(fan::color::parse(v)); });

  loco->gui.console.commands.add("echo", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, gui.console);
    fan::commands_t::output_t out;
    out.text = fan::append_args(args) + "\n";
    out.highlight = fan::graphics::highlight_e::info;
    loco->gui.console.commands.output_cb(out);
  }).description = "prints something - usage echo [args]";

  loco->gui.console.commands.add("help", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, gui.console);
    if (args.empty()) {
      fan::commands_t::output_t out;
      out.highlight = fan::graphics::highlight_e::info;
      std::string out_str;
      out_str += "{\n";
      for (const auto& cmd : loco->gui.console.commands.get_command_list()) {
        out_str += "\t" + cmd.first + ",\n";
      }
      out_str += "}\n";
      out.text = out_str;
      loco->gui.console.commands.output_cb(out);
      return;
    }
    else if (args.size() == 1) {
      if (!loco->gui.console.commands.has_command(args[0])) {
        loco->gui.console.commands.print_command_not_found(args[0]);
        return;
      }
      fan::commands_t::output_t out;
      out.text = loco->gui.console.commands.get_command_description(args[0]) + "\n";
      out.highlight = fan::graphics::highlight_e::info;
      loco->gui.console.commands.output_cb(out);
    }
    else {
      loco->gui.console.commands.print_invalid_arg_count();
    }
  }).description = "get info about specific command - usage help command";

  loco->gui.console.commands.add("list", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, gui.console);
    std::string out_str;
    for (const auto& cmd : loco->gui.console.commands.get_command_list()) {
      out_str += cmd.first + "\n";
    }
    fan::commands_t::output_t out;
    out.text = out_str;
    out.highlight = fan::graphics::highlight_e::info;
    loco->gui.console.commands.output_cb(out);
  }).description = "lists all commands - usage list";

  loco->gui.console.commands.add("alias", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, gui.console);
    if (args.size() < 2 || args[1].empty()) {
      loco->gui.console.commands.print_invalid_arg_count();
      return;
    }
    if (loco->gui.console.commands.insert_to_command_chain(args)) {
      return;
    }
    
    std::string target = args[1];
    loco->gui.console.commands.add(args[0], [target](fan::console_t* self, const fan::commands_t::arg_t& inner_args) {
      std::vector<std::string> new_args = {target};
      new_args.insert(new_args.end(), inner_args.begin(), inner_args.end());
      std::string full_cmd = fan::append_args(new_args);
      self->call(full_cmd);
    }).description = "alias to " + target;
  }).description = "can create alias commands - usage alias [cmd name] [cmd]";

  loco->gui.console.commands.add("quit", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    std::exit(0);
  }).description = "quits program - usage quit";

  loco->gui.console.commands.add("clear", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, gui.console);
    loco->gui.console.call("clear_internal"); // You will need to add a small clear() method to console_t!
  }).description = "clears output buffer - usage clear";

#if defined(FAN_2D)
  loco->gui.console.commands.add("rectangle", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, gui.console);
    if (args.size() < 1 || args.size() > 3) {
      loco->gui.console.commands.print_invalid_arg_count();
      return;
    }
    try {
      fan::graphics::shapes::rectangle_t::properties_t props;
      props.position = fan::vec3::from_string(args[0]);
      if (args.size() >= 2) props.size = fan::vec2::from_string(args[1]);
      props.color = args.size() == 3 ? fan::color::parse(args[2]) : fan::colors::white;

      auto NRI = loco->add_shape_to_static_draw(props);
      loco->gui.console.println_colored("Added rectangle", fan::colors::green);
      loco->gui.console.println(
        "  id: " + std::to_string(NRI) +
        "\n  position " + (std::string)props.position +
        "\n  size " + (std::string)props.size +
        "\n  color " + props.color.to_string(),
        fan::graphics::highlight_e::info
      );
    }
    catch (const std::exception& e) {
      loco->gui.console.println_colored("Invalid arguments: " + std::string(e.what()), fan::colors::red);
    }
  }).description = "Adds static rectangle {x,y[,z]} {w,h} [{r,g,b,a}]";

  loco->gui.console.commands.add("remove_shape", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, gui.console);
    if (args.size() != 1) {
      loco->gui.console.commands.print_invalid_arg_count();
      return;
    }
    try {
      std::uint32_t shape_id = std::stoull(args[0]);
      fan::graphics::shapes::shape_t* s = reinterpret_cast<fan::graphics::shapes::shape_t*>(&shape_id);
      loco->remove_static_shape_draw(*s);
      loco->gui.console.println_colored(
        "Removed shape with id " + std::to_string(shape_id),
        fan::colors::green
      );
    }
    catch (const std::exception& e) {
      loco->gui.console.println_colored("Invalid argument: " + std::string(e.what()), fan::colors::red);
    }
  }).description = "Removes a shape by its id";
#endif

  loco->gui.console.commands.add("print", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, gui.console);
    fan::commands_t::output_t out;
    out.text = fan::append_args(args) + "\n";
    out.highlight = fan::graphics::highlight_e::info;
    loco->gui.text_logger.print(fan::graphics::highlight_color_table[out.highlight], out.text);
  }).description = "prints something to bottom left of screen - usage print [args]";
  
  loco->gui.console.commands.add("dump_dbg", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    static fan::console_t::frame_cb_nr_t active_poll_nr = -1;
    static std::uint64_t log_cursor = 0;
    static constexpr f32_t text_dim = 1.5f;
    auto print_logs = [self](const std::vector<fan::log_entry_t>& logs) {
      for (const auto& log : logs) {
        switch (log.level) {
          case fan::log_level_e::error:
            self->print_colored("[ERROR] ", fan::colors::red / text_dim);
            break;
          case fan::log_level_e::warning:
            self->print_colored("[WARN] ", fan::colors::yellow / text_dim);
            break;
          default:
            self->print_colored("[INFO] ", fan::colors::orange / text_dim);
            break;
          }
        if (log.tag.size()) {
          self->print_colored(std::format("[{}] ", log.tag), fan::colors::cyan / text_dim);
        }
        self->println_colored(log.msg, fan::colors::white / 1.2f);
      }
    };

    if (args.empty() || args[0] == "1") {
      auto logs = fan::dump_memory_logs();
      if (logs.empty()) {
        self->println_colored("Debug buffer is empty.", fan::colors::yellow / text_dim);
        return;
      }
      self->println_colored("--- DEBUG BUFFER DUMP ---", fan::colors::green / text_dim);
      print_logs(logs);
    } 
    else if (args[0] == "2" || args[0] == "on") {
      if (active_poll_nr != static_cast<fan::console_t::frame_cb_nr_t>(-1)) {
        self->println_colored("Active debug printing is already ON.", fan::colors::yellow / text_dim);
        return;
      }
      
      fan::dump_memory_logs_since(log_cursor);
      self->println_colored("Active debug printing ENABLED. Type 'dump_dbg 0' to stop.", fan::colors::green / text_dim);
      
      active_poll_nr = self->push_frame_process([print_logs]() {
        print_logs(fan::dump_memory_logs_since(log_cursor));
      });
    }
    else if (args[0] == "0" || args[0] == "off") {
      if (active_poll_nr != static_cast<fan::console_t::frame_cb_nr_t>(-1)) {
        self->erase_frame_process(active_poll_nr);
        self->println_colored("Active debug printing DISABLED.", fan::colors::yellow / text_dim);
      }
    }
  }).description = "Dumps debug buffer. Usage: dump_dbg [empty=dump once] [2=live feed on] [0=live feed off]";

#endif
}

#if defined(FAN_2D)

void loco_t::culling_rebuild_grid() {
  fan::graphics::culling::static_grid_init(((fan::graphics::culling::culling_t*)shapes.visibility)->static_grid, world_min, cell_size, grid_size);
  fan::graphics::culling::dynamic_grid_init(((fan::graphics::culling::culling_t*)shapes.visibility)->dynamic_grid, world_min, cell_size, grid_size);
}

void loco_t::rebuild_static_culling() {
  fan::graphics::culling::rebuild_static(*((fan::graphics::culling::culling_t*)shapes.visibility));
}

bool loco_t::culling_enabled() const {
  return ((fan::graphics::culling::culling_t*)shapes.visibility)->enabled;
}

void loco_t::set_culling_enabled(bool enabled) {
  fan::graphics::culling::set_enabled(*((fan::graphics::culling::culling_t*)shapes.visibility), enabled);
}

void loco_t::get_culling_stats(std::uint32_t& visible, std::uint32_t& culled) const {
  visible = 0;
  std::uint32_t total = 0;
  for (auto const& [cam_id, cam_state] : ((fan::graphics::culling::culling_t*)shapes.visibility)->camera_states) {
    visible += std::count_if(
      cam_state.visible.begin(),
      cam_state.visible.end(),
      [](const auto& pair) { return pair.second == 1; }
    );
    total = std::max<std::uint32_t>(total, cam_state.visible.size());
  }
  culled = (total >= visible) ? (total - visible) : 0;
}

void loco_t::run_culling() {
  auto& culling = *((fan::graphics::culling::culling_t*)shapes.visibility);
  for_each_list(camera_list, [&](auto&, auto nr) {
    if (nr == perspective_render_view.camera) {
      return;
    }
    fan::graphics::culling::cull_camera(culling, fan::graphics::g_shapes->shaper, nr);
  });
}

void loco_t::set_cull_padding(const fan::vec2& padding) {
  ((fan::graphics::culling::culling_t*)shapes.visibility)->padding = padding;
}

void loco_t::visualize_culling() {
  const auto& cam = camera_get();

  fan::vec2 top_left = fan::graphics::screen_to_world(fan::vec2(0, 0), orthographic_render_view);
  fan::vec2 bottom_right = fan::graphics::screen_to_world(viewport_get_size(), orthographic_render_view);

  fan::vec2 scaled_padding = ((fan::graphics::culling::culling_t*)shapes.visibility)->padding / cam.zoom;
  fan::vec2 padded_min = top_left - scaled_padding;
  fan::vec2 padded_max = bottom_right + scaled_padding;

  fan::vec2 top_right(padded_max.x, padded_min.y);
  fan::vec2 bottom_left(padded_min.x, padded_max.y);

  auto emit_line = [&](fan::vec2 src, fan::vec2 dst) {
    add_shape_to_immediate_draw(fan::graphics::shapes::shape_t(
      fan::graphics::shapes::line_t::properties_t {
        .src = src,
        .dst = dst,
        .color = fan::color(1, 0, 0, 0.8f),
        .thickness = 5.f / cam.zoom
      }, false));
  };

  emit_line(padded_min, top_right);
  emit_line(top_right, padded_max);
  emit_line(padded_max, bottom_left);
  emit_line(bottom_left, padded_min);
}
#endif

void loco_t::check_vk_result(int err) {
  if (err != VK_SUCCESS) {
    fan::print_impl("vkerr", (int)err);
  }
}

#if defined(FAN_GUI)
void loco_t::init_gui() {
  if (::gui::is_gui_initialized()) {
    gui.gui_initialized = true;
    return;
  }
  ::gui::init(
    window,
    context.vk.instance,
    context.vk.physical_device,
    context.vk.device,
    context.vk.queue_family,
    context.vk.graphics_queue,
    context.vk.descriptor_pool,
    context.vk.MainWindowData.RenderPass,
    context.vk.MainWindowData.ImageCount,
    context.vk.MinImageCount,
    VK_SAMPLE_COUNT_1_BIT,
    check_vk_result
  );
  gui.font_future = std::async(std::launch::async, []() {
    ::gui::init_fonts();
  });
  gui.gui_initialized = true;
}

void loco_t::destroy_gui() {
  if (!gui.gui_initialized || !::gui::is_gui_initialized()) {
    return;
  }

  context.vk.gui_close();

  ::gui::shutdown_graphics_context(
    context.vk.device
  );

  context.vk.gui_close_finish();

  ::gui::destroy();
  gui.gui_initialized = false;
}
#endif

void loco_t::bind_global_context() {
  auto& ctx = fan::graphics::ctx();
  ctx.context_functions = &context_functions;
  ctx.render_context = &context;
  ctx.camera_list = &camera_list;
  ctx.shader_list = &shader_list;
  ctx.image_list = &image_list;
  ctx.viewport_list = &viewport_list;
  ctx.window = &window;
  ctx.orthographic_render_view = &orthographic_render_view;
  ctx.perspective_render_view = &perspective_render_view;
  ctx.update_callback = &m_update_callback;
  ctx.lighting = &renderer_state.lighting;
  IF_GUI(ctx.console = &gui.console;)
  IF_GUI(ctx.text_logger = &gui.text_logger;)
}

static void loco_init_shapes_context(loco_t* l) {
#if defined(FAN_2D)
  l->shapes.texture_pack = &l->texture_pack;
  l->shapes.immediate_render_list = &l->immediate_render_list;
  l->shapes.static_render_list = &l->static_render_list;
  l->shapes.shapes_init_pools(&l->shapes);
  fan::graphics::g_shapes = &l->shapes;
  l->shapes.visibility = new fan::graphics::culling::culling_t;
  l->shapes.immediate_shape_caches.resize(fan::graphics::shape_type_t::last);
#endif
}

static void loco_init_platform(loco_t* l) {
#if defined(fan_platform_windows)
  SetConsoleOutputCP(CP_UTF8);
#endif
}

static void loco_init_renderer(loco_t* l) {
  l->get_render_shapes_top() = l->open_props.render_shapes_top;
}

static void loco_load_settings_into_open_props(loco_t* l) {
#if defined(FAN_GUI)
  auto* sm = get_smenu(l);
  if (sm->config.display.custom_resolution.x != -1 && l->open_props.window_size.x == -1)
    l->open_props.window_size = sm->config.display.custom_resolution;
  else if (sm->config.display.resolution_index != -1 && l->open_props.window_size.x == -1)
    l->open_props.window_size = fan::resolutions[sm->config.display.resolution_index].size;
  if (sm->config.display.display_mode != fan::window_t::mode::windowed)
    l->open_props.window_open_mode = sm->config.display.display_mode;
  if (sm->config.display.window_position.x != -1)
    l->open_props.window_position = sm->config.display.window_position;
#endif
}

static void loco_open_window(loco_t* l) {
  l->window.set_antialiasing(l->open_props.samples);
  l->window.open(fan::window_t::properties_t{
    .size = l->open_props.window_size,
    .position = l->open_props.window_position,
    .flags = l->open_props.window_flags,
    .open_mode = l->open_props.window_open_mode
  });
  l->context.vk.enable_clear = !l->get_render_shapes_top();
  l->context.vk.shapes_top = l->get_render_shapes_top();
  l->context.vk.vsync = l->open_props.vsync;
  l->context.vk.open(l->window);
}

static void loco_init_renderer_post_window(loco_t* l) {
  l->start_time.start();
}

static void loco_init_shapes_system(loco_t* l) {
#if defined(FAN_2D)
  l->shapes.shaper.Open();
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::light,        sizeof(std::uint8_t),                              fan::graphics::shaper_t::KeyBitOrderAny);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::light_end,    sizeof(std::uint8_t),                              fan::graphics::shaper_t::KeyBitOrderAny);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::visible,      sizeof(std::uint8_t),                              fan::graphics::shaper_t::KeyBitOrderAny);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::depth,        sizeof(fan::graphics::depth_t),                    fan::graphics::shaper_t::KeyBitOrderLow);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::shader,       sizeof(fan::graphics::shader_raw_t),               fan::graphics::shaper_t::KeyBitOrderLow);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::blending,     sizeof(fan::graphics::blending_t),                 fan::graphics::shaper_t::KeyBitOrderLow);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::image,        sizeof(fan::graphics::image_t),                    fan::graphics::shaper_t::KeyBitOrderAny);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::viewport,     sizeof(fan::graphics::viewport_t),                 fan::graphics::shaper_t::KeyBitOrderAny);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::camera,       sizeof(fan::graphics::camera_t),                   fan::graphics::shaper_t::KeyBitOrderAny);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::ShapeType,    sizeof(fan::graphics::shaper_t::ShapeTypeIndex_t), fan::graphics::shaper_t::KeyBitOrderAny);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::filler,       sizeof(std::uint8_t),                              fan::graphics::shaper_t::KeyBitOrderAny);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::draw_mode,    sizeof(std::uint8_t),                              fan::graphics::shaper_t::KeyBitOrderAny);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::vertex_count, sizeof(std::uint32_t),                             fan::graphics::shaper_t::KeyBitOrderAny);
  fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::shadow,       sizeof(std::uint8_t),                              fan::graphics::shaper_t::KeyBitOrderAny);
#endif
}

static void loco_init_render_views(loco_t* l) {
  fan::vec2 window_size = l->window.get_size();
  l->orthographic_render_view.create_default(window_size);
  l->perspective_render_view.camera = l->open_camera_perspective();
  l->perspective_render_view.viewport = l->open_viewport(fan::vec2(0, 0), window_size);
}

static void loco_init_culling(loco_t* l) {
#if defined(FAN_2D)
  l->cell_size = 256;
  l->culling_rebuild_grid();
  l->set_culling_enabled(true);
#endif
}

static void loco_fire_engine_init_callbacks(loco_t* l) {
  auto it = fan::graphics::get_engine_init_cbs().GetNodeFirst();
  while (it != fan::graphics::get_engine_init_cbs().dst) {
    fan::graphics::get_engine_init_cbs().StartSafeNext(it);
    fan::graphics::get_engine_init_cbs()[it](l);
    it = fan::graphics::get_engine_init_cbs().EndSafeNext();
  }
}

loco_t::loco_t() : loco_t(loco_t::properties_t()) {}

loco_t::loco_t(const loco_t::properties_t& props) :
  open_props(props),
  init_gloco([this] { gloco() = this; return true; }())
{
  fan::time::print_measure("loco_t init start");
  fan::time::timer t_loco;
  fan::time::timer t;
  fan::init_manager_t::cleaner();
  fan::time::measure(t, "cleaner");
  fan::event::init_dispatcher();
  fan::time::measure(t, "init_dispatcher");

  idle_handle  = new fan::uv::idle_t;
  timer_handle = new fan::uv::timer_t;
  
  #if defined(FAN_GUI)
  {
    auto& ctx = fan::graphics::ctx();
    ctx.input_action = &input.input_action;
    gui.settings_menu = new ::gui::settings_menu_t;
  }
  #endif
  fan::time::measure(t, "gui settings");

  bind_global_context();
  fan::time::measure(t, "bind_global_context");
  loco_init_shapes_context(this);
  fan::time::measure(t, "loco_init_shapes_context");
  loco_init_platform(this);
  fan::time::measure(t, "loco_init_platform");
  loco_init_renderer(this);
  fan::time::measure(t, "loco_init_renderer");
  loco_load_settings_into_open_props(this);
  fan::time::measure(t, "loco_load_settings_into_open_props");

#if defined(FAN_AUDIO)
  std::jthread audio_init_thread([this] { audio.init(); });
#endif

  context_functions = fan::graphics::get_vk_context_functions();
  new (&context.vk) fan::vulkan::context_t();

#if defined(FAN_2D)
  vk = new loco_t::vulkan_t;
  vk->loco_ptr = this;
  vk->shaders_compile_preload();
#endif

  loco_open_window(this);
  fan::time::measure(t, "loco_open_window");
  loco_init_renderer_post_window(this);
  fan::time::measure(t, "loco_init_renderer_post_window");
  loco_init_shapes_system(this);
  fan::time::measure(t, "loco_init_shapes_system");
  loco_init_render_views(this);
  fan::time::measure(t, "loco_init_render_views");
#if defined(FAN_2D)
  vk->shaders_compile();
  fan::time::measure(t, "vk.shaders_compile");
#endif
  vk->init();
  fan::time::measure(t, "vk.init");
#if defined(FAN_2D)
  load_engine_images();
  fan::time::measure(t, "load_engine_images");
  vk->shapes_open();
  fan::time::measure(t, "vk.shapes_open");
#endif
#if defined(FAN_GUI)
  init_gui();
  fan::time::measure(t, "init_gui");
  flush_startup_async_images();
  generate_commands(this);
  fan::time::measure(t, "generate_commands");
#endif
  input.init(window);
  fan::time::measure(t, "input.init");
  #if defined(FAN_AUDIO)
  if (audio_init_thread.joinable()) {
    audio_init_thread.join();
  }
  fan::time::measure(t, "audio.init wait");
  #endif
  fan::graphics::ctx().default_texture = default_texture;
#if defined(FAN_GUI)
  gui.console.commands.call("debug_memory " + std::to_string((int)fan::memory::heap_profiler_t::instance().enabled));
#endif
   loco_init_culling(this);
  fan::time::measure(t, "loco_init_culling");
#if defined(FAN_GUI)
  get_smenu(this)->init_runtime();
#endif
  loco_fire_engine_init_callbacks(this);

  vkQueueWaitIdle(context.vk.graphics_queue);

  auto shaders_path = fan::io::file::find_relative_path("shaders").generic_string();
  if (shaders_path.empty()) {
    fan::print_error("failed to find path for 'shaders' - hot reloading is disabled for shaders");
  }
  else {
    shader_watcher = new fan::event::fs_watcher_t(shaders_path);
    auto started = shader_watcher->start([this, shaders_path](const std::string& filename, int events) {
      if (fan::io::file::is_temp_file(filename)) { return; }

      std::string normalized = filename;
      std::replace(normalized.begin(), normalized.end(), '\\', '/');

      std::error_code ec;
      std::filesystem::path full_path = std::filesystem::path(shaders_path) / normalized;
      if (std::filesystem::is_directory(full_path, ec) || !std::filesystem::exists(full_path, ec)) { return; }

      for_each_list(shader_list, [&](auto& list, auto nr) {
        auto& shader = list[nr];
        bool reload_needed = false;

        auto check_and_update = [&](auto& path_ct, std::string& src_code, auto set_fn) {
          std::string path(path_ct.c_str());
          if (!path.empty() && (path.ends_with(normalized) || normalized.ends_with(path))) {
            std::string new_code;
            if (!fan::io::file::read(path, &new_code) && !new_code.empty() && new_code != src_code) {
              src_code = std::move(new_code);
              set_fn(nr, path, src_code);
              reload_needed = true;
            }
          }
        };

        check_and_update(shader.path_vertex, shader.svertex, [this](auto n, const auto& p, const auto& c) {
          shader_set_vertex(n, p, c);
        });
        check_and_update(shader.path_fragment, shader.sfragment, [this](auto n, const auto& p, const auto& c) {
          shader_set_fragment(n, p, c);
        });
        check_and_update(shader.path_compute, shader.scompute, [this](auto n, const auto& p, const auto& c) {
          shader_set_compute(n, p, c);
        });

        if (!reload_needed || !shader_compile(nr)) { return; }

        vkDeviceWaitIdle(context.vk.device);
  #if defined(FAN_2D)
        for (auto& st : fan::graphics::g_shapes->shaper.ShapeTypes) {
          if (st.sti == (decltype(st.sti))-1 || st.renderer.vk.pipeline.properties.shader != nr) {
            continue;
          }

          auto props = st.renderer.vk.pipeline.properties;
          st.renderer.vk.pipeline.close(context.vk);

          props.descriptor_layouts.clear();
          if (st.renderer.vk.shape_data.m_descriptor.m_layout != VK_NULL_HANDLE) {
            props.descriptor_layouts.push_back(st.renderer.vk.shape_data.m_descriptor.m_layout);
          }

          props.push_constants_size = sizeof(fan::vulkan::context_t::push_constants_t);

          VkPipelineColorBlendAttachmentState attachment = fan::vulkan::get_default_color_blend();
          if (st.sti == fan::graphics::shapes::shape_type_t::light) {
            attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
          }
          props.color_blend_attachments = { attachment };

          st.renderer.vk.pipeline.open(context.vk, props);
        }
        if (nr == vk->post_process_shader || nr == vk->bloom_downsample_shader || nr == vk->bloom_upsample_shader) {
          vk->close_post_process_pipelines();
          vk->open_post_process_pipelines();
        }
  #endif
      });
    });
    if (!started) {
      fan::print_error("Shader reload watch failed to start:", started.error());
    }
  }
  fan::time::print_measure("loco total took:", t_loco.millis(), "ms");
}

loco_t::loco_t(std::function<void()> loop_fn) : loco_t(loop_fn, properties_t()) {}

loco_t::loco_t(std::function<void()> loop_fn, const properties_t& p) : loco_t(p){
  loop(loop_fn);
}


loco_t::~loco_t() {
  destroy();
}

void loco_t::destroy() {
  if (window == nullptr) {
    return;
  }

  fan::event::close_dispatcher();

  async_image_destroy();

  if (idle_handle) {
    if (idle_init) {
      fan::uv::idle_stop((fan::uv::idle_t*)idle_handle);
      fan::uv::close((fan::uv::handle_t*)idle_handle, [](fan::uv::handle_t* h) {
        delete (fan::uv::idle_t*)h;
      });
    }
    else {
      delete (fan::uv::idle_t*)idle_handle;
    }
    idle_handle = nullptr;
  }
  if (timer_handle) {
    if (timer_init) {
      fan::uv::timer_stop((fan::uv::timer_t*)timer_handle);
      fan::uv::close((fan::uv::handle_t*)timer_handle, [](fan::uv::handle_t* h) {
        delete (fan::uv::timer_t*)h;
      });
    }
    else {
      delete (fan::uv::timer_t*)timer_handle;
    }
    timer_handle = nullptr;
  }

  if (shader_watcher) {
    delete shader_watcher;
    shader_watcher = nullptr;
  }
#if defined(FAN_GUI)
  delete (::gui::settings_menu_t*)gui.settings_menu;
  gui.settings_menu = nullptr;
#endif

  fan::uv::run((fan::uv::loop_t*)fan::event::get_loop(), fan::uv::run_default);

#if defined(FAN_2D)
  static_render_list.clear();
  immediate_render_list.clear();
  shapes.immediate_shape_caches.clear();
  fan::graphics::flush_destruct_callbacks();
  {
    fan::graphics::shapes::shape_ids_t::nrtra_t nrtra;
    fan::graphics::shapes::shape_ids_t::nr_t nr;
    nrtra.Open(&fan::graphics::g_shapes->shape_ids, &nr);
    while (nrtra.Loop(&fan::graphics::g_shapes->shape_ids, &nr)) {
      fan::graphics::g_shapes->remove_shape(nr.NRI);
    }
    nrtra.Close(&fan::graphics::g_shapes->shape_ids);
  }
  shapes.shapes_destroy_pools(&shapes);
  delete ((fan::graphics::culling::culling_t*)shapes.visibility);
  shapes.visibility = nullptr;
#endif

#if defined(FAN_GUI)
  gui.console.commands.get_command_list().clear();
  gui.console.close();
#endif

  vkDeviceWaitIdle(context.vk.device);
#if defined(FAN_2D)
  for (auto& st : fan::graphics::g_shapes->shaper.ShapeTypes) {
    if (st.sti == (decltype(st.sti))-1) {
      continue;
    }
    st.renderer.vk.shape_data.close(context.vk);
    st.renderer.vk.pipeline.close(context.vk);
  }
#endif

  vk->close();
  delete vk;
  vk = nullptr;

#if defined(FAN_2D)
  fan::graphics::g_shapes->shaper.Close();
#endif

#if defined(FAN_GUI)
  destroy_gui();
#else
  context.vk.close();
#endif

  window.close();
#if defined(FAN_AUDIO)
  audio.destroy();
#endif

  fan::event::loop_close();
}

void loco_t::close() {
  window.set_should_close(true);
}

void loco_t::shapes_draw() {
  timing.shape_draw_timer.start();

  fan::time::global_profiler.begin("vk->shapes_draw");
  vk->shapes_draw();
  fan::time::global_profiler.end("vk->shapes_draw");

  timing.shape_draw_time_s = timing.shape_draw_timer.seconds();

#if defined(FAN_2D)
  fan::time::global_profiler.begin("immediate_render_list.clear()");
  immediate_render_list.clear();
  fan::time::global_profiler.end("immediate_render_list.clear()");
  fan::time::global_profiler.begin("immediate_shape_caches.clear()");
  {
    auto& caches = shapes.immediate_shape_caches;
    for (auto& cache : caches) {
      int used = cache.used_this_frame;
      // hide entries beyond this frame's count (stale from prior frames)
      for (int i = used; i < (int)cache.shapes.size(); ++i) {
        cache.shapes[i].set_visible(false);
      }
      cache.used_this_frame = 0;
    }
  }
  fan::time::global_profiler.end("immediate_shape_caches.clear()");
  fan::time::global_profiler.begin("ProcessBlockEditQueue");
  fan::graphics::g_shapes->shaper.ProcessBlockEditQueue();
  fan::time::global_profiler.end("ProcessBlockEditQueue");
#endif
}

void loco_t::process_shapes() {

  if (get_render_shapes_top() == true) {
    fan::time::global_profiler.begin("Begin Render Pass");
    vk->begin_render_pass();
    fan::time::global_profiler.end("Begin Render Pass");
  }

  fan::time::global_profiler.begin("Pre Draw Callbacks");
  for (const auto& i : m_pre_draw) {
    i();
  }
  fan::time::global_profiler.end("Pre Draw Callbacks");

  fan::time::global_profiler.begin("Shapes Draw");
  shapes_draw();
  fan::time::global_profiler.end("Shapes Draw");

  fan::time::global_profiler.begin("Post Draw Callbacks");
  for (const auto& func : m_post_draw) {
    func();
  }
  fan::time::global_profiler.end("Post Draw Callbacks");

  if (vk->image_error == VK_SUCCESS) {
    fan::time::global_profiler.begin("Draw Post Process");
    vk->draw_post_process();
    fan::time::global_profiler.end("Draw Post Process");
  }
}

void loco_t::process_gui() {
  using namespace fan::graphics;
  timing.gui_draw_timer.start();
#if defined(FAN_GUI)
  ::gui::process_frame();

  if (auto h = ::gui::hud("##text_logger")) {
    gui.text_logger.render();
  }

  if (input.input_action.is_clicked(fan::actions::toggle_console)) {
    gui.render_console = !gui.render_console;
    gui.console.force_focus();
  }
  if (gui.render_console) {
    gui.console.render();
  }
  if (!get_smenu(this)->keybind_menu.is_capturing() &&
    input.input_action.is_clicked(fan::actions::toggle_settings) &&
    !get_smenu(this)->keybind_menu.should_suppress_input()) {
    if (gui.render_console) {
      gui.render_console = false;
    }
    else {
      gui.render_settings_menu = !gui.render_settings_menu;
    }
  }
  get_smenu(this)->update();
  if (gui.render_settings_menu) {
    get_smenu(this)->render();
  }

  if (gui.render_debug_memory) {
    ::gui::set_next_window_bg_alpha(0.99f);
    ::gui::window_flags_t window_flags = ::gui::window_flags_no_title_bar | ::gui::window_flags_topmost;
    ::gui::set_next_window_size(fan::vec2(950, 750), ::gui::cond_once);
    ::gui::begin("fan_memory_dbg_wnd", nullptr, window_flags);
    ::gui::render_allocations_plot();
    ::gui::end();
  }

  if (gui.show_fps) {
    ::gui::window_flags_t window_flags = ::gui::window_flags_no_title_bar | ::gui::window_flags_topmost;

    ::gui::set_next_window_bg_alpha(0.99f);
    ::gui::set_next_window_size(fan::vec2(831.0000, 693.0000), ::gui::cond_once);
    ::gui::begin("Performance window", nullptr, window_flags);

    gui.frame_monitor.update(get_delta_time());
    gui.shape_monitor.update(timing.shape_draw_time_s);
    gui.gui_monitor.update(timing.gui_draw_time_s);

    auto frame_stats = gui.frame_monitor.stats();
    auto shape_stats = gui.shape_monitor.stats();
    auto gui_stats = gui.gui_monitor.stats();

    static auto format_val = [](double v, int prec = 4) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(prec) << v;
      return oss.str();
    };

    ::gui::text("Current FPS:", std::to_string(static_cast<int>(1.f / get_delta_time())));
    ::gui::text("Average FPS:", std::to_string(static_cast<int>(frame_stats.avg_fps())));
    ::gui::text("Lowest FPS:", std::to_string(static_cast<int>(frame_stats.min_fps())));
    ::gui::text("Highest FPS:", std::to_string(static_cast<int>(frame_stats.max_fps())));
    ::gui::text("Frame Time Avg:", format_val(frame_stats.avg_frame_time_s * 1e3) + " ms");
    ::gui::text("Shape Draw Avg:", format_val(shape_stats.avg_frame_time_s * 1e3) + " ms");
    ::gui::text("GUI Draw Avg:", format_val(gui_stats.avg_frame_time_s * 1e3) + " ms");

    if (::gui::button(gui.frame_monitor.paused ? "Continue" : "Pause")) {
      gui.frame_monitor.paused = !gui.frame_monitor.paused;
      gui.shape_monitor.paused = gui.frame_monitor.paused;
      gui.gui_monitor.paused = gui.frame_monitor.paused;
    }

    if (::gui::button("Reset data")) {
      gui.frame_monitor.reset();
      gui.shape_monitor.reset();
      gui.gui_monitor.reset();
    }

    if (::gui::plot::begin_plot("Times", fan::vec2(-1, 0), ::gui::plot::flags_no_frame)) {
      ::gui::plot::setup_axes("Frame Index", "Frame Time (ms)",
        ::gui::plot::axis_flags_auto_fit,
        ::gui::plot::axis_flags_auto_fit | ::gui::plot::axis_flags_range_fit
      );
      ::gui::plot::setup_axis_ticks(::gui::plot::axis_y1, 0.0, 10.0, 11);
      gui.frame_monitor.plot(this, "Frame Draw Time");
      gui.shape_monitor.plot(this, "Shape Draw Time");
      gui.gui_monitor.plot(this, "GUI Draw Time");

      if (gui.frame_monitor.buffer.size() > gui.time_plot_scroll.view_size) {
        int max_offset = static_cast<int>(gui.frame_monitor.buffer.size()) - gui.time_plot_scroll.view_size;
        ::gui::drag("Scroll", &gui.time_plot_scroll.scroll_offset, 1.f, 0.f, max_offset, ::gui::slider_flags_always_clamp);
      }
      ::gui::plot::end_plot();
    }

    ::gui::text("Frame Draw Time: ", format_val(get_delta_time() * 1e3) + " ms");
    ::gui::text("Shape Draw Time: ", format_val(timing.shape_draw_time_s * 1e3) + " ms");
    ::gui::text("GUI Draw Time: ", format_val(timing.gui_draw_time_s * 1e3) + " ms");

    ::gui::end();
  }

  {
    gui.frame_count++;

    if (!gui.fps_timer.started()) {
      gui.last_fps = 1.0 / get_delta_time();
      gui.fps_timer.start_seconds(1.0f);
    }

    if (gui.fps_timer.finished()) {
      gui.last_fps = gui.frame_count;
      gui.frame_count = 0;
      gui.fps_timer.restart();
    }

    std::string s = std::to_string(gui.last_fps);

    ::gui::push_font(::gui::get_font(15.f));

    fan::vec2 ts = ::gui::calc_text_size(s);
    fan::vec2 bs = fan::vec2(34, ts.y * 1.4f);
    fan::vec2 p = fan::vec2(window.get_size().x - bs.x, 0);
    fan::vec2 tp = fan::vec2(p.x + (bs.x - ts.x) * 0.5f, p.y + (bs.y - ts.y) * 0.5f);

    auto* dl = ::gui::get_foreground_draw_list();
    auto bg = fan::color(0, 0, 0, 1).get_gui_color();
    auto border = ::gui::get_color_u32(::gui::col_border);
    f32_t rounding = ::gui::get_style().FrameRounding;

    dl->AddRectFilled(fan::vec2(p.x, p.y), fan::vec2(p.x + bs.x, p.y + bs.y), bg, rounding);
    dl->AddRect(fan::vec2(p.x, p.y), fan::vec2(p.x + bs.x, p.y + bs.y), border, rounding);
    dl->AddText(fan::vec2(tp.x, tp.y), fan::color(0, 1, 0, 1).get_gui_color(), s.c_str());

    ::gui::pop_font();
  }

  ::gui::enforce_topmost();

  ::gui::render(
    get_render_shapes_top(),
    &get_vk_context(),
    renderer_state.clear_color,
    vk->image_error,
    context.vk.command_buffers[context.vk.current_frame],
    fan::vulkan::context_t::ImGuiFrameRender
  );
#endif
#if defined(FAN_GUI)
  ::gui::set_want_io();
#endif
  timing.gui_draw_time_s = timing.gui_draw_timer.seconds();
}

void loco_t::get_vram_usage(int* total_mem_MB, int* used_MB) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(context.vk.physical_device, &mem_props);

  std::vector<VmaBudget> budgets(mem_props.memoryHeapCount);
  vmaGetHeapBudgets(context.vk.allocator, budgets.data());

  VkDeviceSize total_bytes = 0;
  VkDeviceSize used_bytes = 0;
  for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
    if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total_bytes += budgets[i].budget;
      used_bytes += budgets[i].usage;
    }
  }
  *total_mem_MB = static_cast<int>(total_bytes / (1024 * 1024));
  *used_MB = static_cast<int>(used_bytes / (1024 * 1024));
}

void loco_t::process_render() {
#if defined(FAN_2D)
  if (init_culling) {
    rebuild_static_culling();
    init_culling = false;
  }
#endif


#if defined(FAN_2D)
  if (is_visualizing_culling) {
    visualize_culling();
  }
  fan::time::global_profiler.begin("Render: Culling");
  run_culling();
  fan::time::global_profiler.end("Render: Culling");
#endif

#if defined(FAN_GUI)
  ::gui::end();
#endif


#if defined(FAN_2D)

  fan::time::global_profiler.begin("Render: Begin Draw");
  vk->begin_draw();
  fan::time::global_profiler.end("Render: Begin Draw");

  fan::time::global_profiler.begin("Render: Memory Q");
  context.vk.memory_queue.process(context.vk);
  fan::time::global_profiler.end("Render: Memory Q");

  fan::time::global_profiler.begin("Render: Block Edit Q");
  fan::graphics::g_shapes->shaper.ProcessBlockEditQueue();
  fan::time::global_profiler.end("Render: Block Edit Q");

  viewport_set(0, window.get_size());

  fan::time::global_profiler.begin("Render: Process Shapes");
  if (get_render_shapes_top() == false) {
    process_shapes();
    fan::time::global_profiler.end("Render: Process Shapes");
    fan::time::global_profiler.begin("Render: Process GUI");
    process_gui();
    fan::time::global_profiler.end("Render: Process GUI");
  }
  else {
    process_gui();
    fan::time::global_profiler.end("Render: Process Shapes"); // Wait, shapes and gui are flipped here, I'll measure them together or separately accurately.
  }

  for (auto& i : draw_end_cb) {
    i();
  }

  if (vk->image_error != VK_SUCCESS) {
    context.vk.command_buffer_in_use = false;
  }
  else {
    fan::time::global_profiler.begin("Render: End Render");
    VkResult err = context.vk.end_render(&window);
    fan::time::global_profiler.end("Render: End Render");
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR || context.vk.SwapChainRebuild) {
      vk->close_swapchain_resources();
      context.vk.recreate_swap_chain(&window, err);
      vk->open_swapchain_resources();
    }
  }
}

bool loco_t::should_close() {
  if (window == nullptr) {
    return true;
  }
  return window.should_close();
}


bool loco_t::process_frame() {
  return process_frame([](f32_t) { });
}

bool loco_t::process_frame(const std::function<void()>& cb) {
  return process_frame([cb](f32_t) { cb(); });
}

bool loco_t::process_frame(const std::function<void(f32_t delta_time)>& cb) {
  fan::time::global_profiler.enabled = get_smenu(this) && get_smenu(this)->config.performance.show_profiler;
  fan::time::global_profiler.update();
  fan::time::global_profiler.begin("Frame Total CPU");
  fan::time::global_profiler.begin("Events");
  window.handle_events();
  time = start_time.seconds();

  if (should_close()) {
    return 1;
  }

  get_delta_time() = window.m_delta_time;

  fan::time::global_profiler.end("Events");

  process_async_image_uploads();

#if defined(FAN_PHYSICS_2D)
  fan::time::global_profiler.begin("Physics");
  physics.context.begin_frame(get_delta_time());
  fan::time::global_profiler.end("Physics");
#endif

  renderer_state.lighting.update(get_delta_time());
  if (get_smenu(this)->config.post_processing.ambient_color != renderer_state.lighting.target) {
    get_smenu(this)->config.post_processing.ambient_color = renderer_state.lighting.target;
  }

#if defined(FAN_GUI)
  if (gui.font_future.valid()) {
    gui.font_future.get();
    gui.font_future = {};
  }

  ::gui::new_frame();
  ::gui::gizmo::begin_frame();

  if (get_smenu(this)->config.performance.show_profiler) {
    ::gui::begin("Engine Profiler", &get_smenu(this)->config.performance.show_profiler, ::gui::window_flags_topmost);
    
    auto render_profiler_node = [](auto& self, const fan::time::profiler_t::entry_t& node) -> void {
      std::string text = std::string(node.name) + ": " + std::to_string(node.last_average) + " ms###" + std::string(node.name);
      if (node.children.empty()) {
        ::gui::tree_node_ex(text, ::gui::tree_node_flags_leaf | ::gui::tree_node_flags_no_tree_push_on_open);
      } else {
        bool open = ::gui::tree_node_ex(text, ::gui::tree_node_flags_default_open);
        if (open) {
          ::gui::indent(15.0f);
          for (const auto& pair : node.children) {
            self(self, pair.second);
          }
          ::gui::unindent(15.0f);
          ::gui::tree_pop();
        }
      }
    };

    for (const auto& pair : fan::time::global_profiler.roots) {
      render_profiler_node(render_profiler_node, pair.second);
    }
    ::gui::end();
  }


  using namespace fan::graphics;

  ::gui::push_style_color(::gui::col_window_bg, fan::color(0, 0, 0, 0));
  ::gui::push_style_color(::gui::col_docking_empty_bg, fan::color(0, 0, 0, 0));
  ::gui::dock_space_over_viewport(0, ::gui::get_main_viewport());

  if (gui.allow_docking || is_key_down(fan::key_left_control)) {
    ::gui::get_io().ConfigFlags |= ::gui::config_flags_docking_enable;
  }
  else {
    ::gui::get_io().ConfigFlags &= ~::gui::config_flags_docking_enable;
  }

  ::gui::pop_style_color(2);

  ::gui::set_next_window_pos(fan::vec2(0, 0));
  ::gui::set_next_window_size(fan::vec2(window.get_size()));

  {
    static constexpr int wnd_flags =
      ::gui::window_flags_no_docking | ::gui::window_flags_no_saved_settings |
      ::gui::window_flags_no_focus_on_appearing | ::gui::window_flags_no_move |
      ::gui::window_flags_no_collapse | ::gui::window_flags_no_background |
      ::gui::window_flags_no_resize | ::gui::dock_node_flags_no_docking_split |
      ::gui::window_flags_no_title_bar | ::gui::window_flags_no_bring_to_front_on_focus |
      ::gui::window_flags_no_inputs
      ;
    ::gui::begin("##global_renderer", nullptr, wnd_flags | (!gui.enable_overlay ? ::gui::window_flags_no_nav | ::gui::window_flags_override_input : 0));
  }
#endif

  fan::time::global_profiler.begin("Events");
  fan::event::process();
  fan::time::process_tasks();
  fan::time::global_profiler.end("Events");

  for (const auto& i : single_queue) {
    i();
  }
  single_queue.clear();

  std::vector<std::coroutine_handle<>> current_frame;
  std::swap(current_frame, fan::graphics::next_frame_awaiter::get_pending());
  for (const auto& h : current_frame) {
    h.resume();
  }

#if defined(FAN_PHYSICS_2D)
  fan::time::global_profiler.begin("Physics");
  physics.update(get_delta_time());
  fan::time::global_profiler.end("Physics");

  if (input.input_action.is_clicked(fan::actions::toggle_debug_physics)) {
    fan::physics::debug_draw_cb()(!fan::physics::is_debug_draw_enabled(), &fan::graphics::get_orthographic_render_view());
  }
#endif
  #if defined(FAN_2D)
  if (input.input_action.is_toggled(fan::actions::toggle_debug_light_buffer)) {
    debug_draw_light_buffer();
  }
  if (input.input_action.is_clicked(fan::actions::recompile_shaders)) {
    shader_recompile_all();
  }
  #endif

  {
    auto it = m_update_callback.GetNodeFirst();
    while (it != m_update_callback.dst) {
      m_update_callback.StartSafeNext(it);
      m_update_callback[it](this);
      it = m_update_callback.EndSafeNext();
    }
  }

  async_image_process();

  fan::time::global_profiler.begin("Game Logic");
  cb(get_delta_time());
  fan::time::global_profiler.end("Game Logic");

#if defined(FAN_PHYSICS_2D)
  physics.draw();
#endif

#if defined(FAN_2D)
  if (renderer_state.force_line_draw) {
    // vulkan draw_all_shape_aabbs
  }
#endif

  if (should_close()) {
    return 1;
  }

  fan::time::global_profiler.begin("CPU Render Submit");
  process_render();
  fan::time::global_profiler.end("CPU Render Submit");
  fan::time::global_profiler.end("Frame Total CPU");

  return 0;
}

void loco_t::loop() {
  loop([&](f32_t) { });
}

void loco_t::loop(const std::function<void()>& cb) {
  loop([&](f32_t) { cb(); });
}

void loco_t::loop(const std::function<void(f32_t delta_time)>& cb) {
  main_loop = cb;
g_loop:

  if (!timer_init) {
    fan::uv::timer_init((fan::uv::loop_t*)fan::event::get_loop(), (fan::uv::timer_t*)timer_handle);
    timer_init = true;
  }
  if (!idle_init) {
    fan::uv::idle_init ((fan::uv::loop_t*)fan::event::get_loop(), (fan::uv::idle_t*)idle_handle);
    idle_init = true;
  }

  ((fan::uv::timer_t*)timer_handle)->data = this;
  ((fan::uv::idle_t*)idle_handle)->data   = this;

  if (target_fps > 0) {
    start_timer();
  }
  else {
    start_idle();
  }

  fan::event::loop();
  if (should_close() == false) {
    goto g_loop;
  }
}

loco_t::camera_t loco_t::open_camera(const fan::vec2& x, const fan::vec2& y) {
  loco_t::camera_t camera = camera_create();
  camera_set_ortho(camera, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
  return camera;
}

loco_t::camera_t loco_t::open_camera_perspective(f32_t fov, f32_t znear, f32_t zfar) {
  loco_t::camera_t camera = camera_create();
  auto& cam = camera_get(camera);
  cam.znear = znear;
  cam.zfar = zfar;
  camera_set_perspective(camera, fov, window.get_size());
  return camera;
}

fan::graphics::viewport_t loco_t::open_viewport(const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
  fan::graphics::viewport_t viewport = viewport_create();
  viewport_set(viewport, viewport_position, viewport_size);
  return viewport;
}

void loco_t::set_viewport(fan::graphics::viewport_t viewport, const fan::vec2& viewport_position, const fan::vec2& viewport_size) {
  viewport_set(viewport, viewport_position, viewport_size);
}

fan::vec2 loco_t::get_input_vector(
  const std::string& forward,
  const std::string& back,
  const std::string& left,
  const std::string& right
) {
  auto& ia = *fan::graphics::ctx().input_action;
  fan::vec2 v(
    ia.is_down(right) - ia.is_down(left),
    ia.is_down(back) - ia.is_down(forward)
  );
  fan::vec2 v2 = fan::graphics::ctx().window->get_gamepad_axis(fan::graphics::ctx().input_action->get_first_gamepad_key(left));
  if (v2.length() > fan::graphics::ctx().window->gamepad_axis_deadzone) {
    return v2;
  }
  return v;
}

fan::vec2 loco_t::get_input_vector(fan::vec2 scalar) {
  return get_input_vector() * scalar;
}

fan::vec2 loco_t::transform_matrix(const fan::vec2& position) {
  fan::vec2 window_size = window.get_size();
  // not custom ortho friendly - made for -1 1
  return position / window_size * 2 - 1;
}

fan::vec2 loco_t::screen_to_ndc(const fan::vec2& screen_pos) {
  return screen_pos / window.get_size() * 2 - 1;
}

fan::vec2 loco_t::ndc_to_screen(const fan::vec2& ndc_position) {
  return (ndc_position + 1) / 2 * window.get_size();
}

void loco_t::set_vsync(bool flag) {
  timing.vsync = flag;
  // vulkan vsync is enabled by presentation mode in swap chain
  fan::graphics::get_vk_context().set_vsync(&window, flag);
}

void loco_t::start_timer() {
  if (target_fps <= 0) return;

  std::uint64_t delay = target_fps > 60 ? 1 : std::max(1.0, std::floor(1.0 / target_fps * 1000.0 * 0.5));

  fan::uv::timer_start((fan::uv::timer_t*)timer_handle, [](fan::uv::timer_t* handle) {
    loco_t* loco = static_cast<loco_t*>(handle->data);

    f64_t elapsed = loco->timing.frame_timer.seconds();
    loco->timing.frame_timer.restart();

    if (elapsed > 0.1) elapsed = 0.1;

    loco->timing.accumulated_time += elapsed;

    if (loco->timing.accumulated_time >= loco->timing.target_frame_time) {
      loco->timing.accumulated_time -= loco->timing.target_frame_time;

      if (loco->process_frame(loco->main_loop)) {
        fan::uv::timer_stop((fan::uv::timer_t*)handle);
        fan::uv::stop((fan::uv::loop_t*)fan::event::get_loop());
      }
    }
  }, delay, delay);
}

void loco_t::idle_cb(void* handle) {
  loco_t* loco = static_cast<loco_t*>(((fan::uv::idle_t*)handle)->data);
  if (loco->process_frame(loco->main_loop)) {
    fan::uv::idle_stop((fan::uv::idle_t*)handle);
    fan::uv::stop((fan::uv::loop_t*)fan::event::get_loop());
  }
}

void loco_t::start_idle(bool start_idle) {
  if (!start_idle) {
    return;
  }
  fan::uv::idle_start((fan::uv::idle_t*)idle_handle, (fan::uv::idle_cb)idle_cb);
}

// if target fps does not seem to be accurate/updating, use timeBeginPeriod to request the correct hz for libuv
void loco_t::update_timer_interval(bool idle) {
  if (!timer_init) {
    fan::uv::timer_init((fan::uv::loop_t*)fan::event::get_loop(), (fan::uv::timer_t*)timer_handle);
    timer_init = true;
  }
  if (!idle_init) {
    fan::uv::idle_init((fan::uv::loop_t*)fan::event::get_loop(), (fan::uv::idle_t*)idle_handle);
    idle_init = true;
  }

  if (idle_init) {
    fan::uv::idle_stop((fan::uv::idle_t*)idle_handle);
  }
  if (timer_init) {
    fan::uv::timer_stop((fan::uv::timer_t*)timer_handle);
  }

  if (target_fps > 0) {
    timing.target_frame_time = 1.0 / target_fps;
    timing.frame_timer.start();
    timing.accumulated_time = 0.0;
    timing.timer_enabled = true;
    start_timer();
  }
  else {
    timing.timer_enabled = false;
    if (idle_init && idle) {
      fan::uv::idle_start((fan::uv::idle_t*)idle_handle, (fan::uv::idle_cb)idle_cb);
    }
  }
}

void loco_t::set_target_fps(std::int32_t new_target_fps, bool idle) {
  target_fps = new_target_fps;
  update_timer_interval(idle);
}

fan::graphics::context_t& loco_t::get_context() {
  return context;
}

fan::graphics::render_view_t loco_t::render_view_create() {
  fan::graphics::render_view_t render_view;
  render_view.create();
  return render_view;
}

fan::graphics::render_view_t loco_t::render_view_create(
  const fan::vec2& ortho_x, const fan::vec2& ortho_y,
  const fan::vec2& viewport_position, const fan::vec2& viewport_size
) {
  fan::graphics::render_view_t render_view;
  render_view.create();
  render_view.set(ortho_x, ortho_y, viewport_position, viewport_size, window.get_size());
  return render_view;
}

loco_t::update_callback_handle_t loco_t::add_update_callback(std::function<void(void*)>&& cb) {
  loco_t::update_callback_handle_t it = m_update_callback.NewNodeLast();
  m_update_callback[it] = std::move(cb);
  return it;
}

loco_t::update_callback_handle_t loco_t::add_update_callback_front(std::function<void(void*)>&& cb) {
  loco_t::update_callback_handle_t it = m_update_callback.NewNodeFirst();
  m_update_callback[it] = std::move(cb);
  return it;
}

void loco_t::remove_update_callback(update_callback_handle_t handle) {
  m_update_callback.unlrec(handle);
}

void loco_t::load_engine_images() {
  default_texture = create_missing_texture();

  fan::graphics::icons.play = request_image_load_async("icons/play.png");
  fan::graphics::icons.pause = request_image_load_async("icons/pause.png");
  fan::graphics::icons.settings = request_image_load_async("icons/settings.png");

  fan::graphics::tile_world_images.dirt = fan::color::from_rgb(0x492201);
  fan::graphics::tile_world_images.background = fan::color::from_rgb(0x20a7db);
}

void loco_t::set_window_name(const std::string& name) {
  window.set_name(name);
}

void loco_t::set_window_icon(const fan::image::info_t& info) {
  window.set_icon(info);
}

void loco_t::set_window_icon(const fan::graphics::image_t& image) {
  auto& image_data = image_list[image];
  auto image_pixels = image_get_pixel_data(image, image_data.image_settings.format);
  fan::image::info_t info;
  info.size = image_data.size;
  info.data = image_pixels.data();
  window.set_icon(info);
}

#if defined(FAN_2D)
void loco_t::debug_draw_light_buffer() {
  fan::vec2 window_size = window.get_size();
  fan::vec2 cam_pos = camera_get_position();
  fan::vec2 viewport_size = viewport_get_size();
  fan::graphics::shapes::rectangle_t::properties_t r;
  { // bg
    r.position = fan::vec3(window_size * 0.5f + cam_pos - viewport_size / 2.f, 65532);
    r.size = window_size * 0.5f;
    r.color = fan::color::rgb(0, 255, 0, 55.f);
    add_shape_to_immediate_draw(fan::graphics::shape_t(r, false));
  }
}
#endif

#if defined(FAN_PHYSICS_2D)
void loco_t::update_physics(bool flag) {
  physics.set_enabled(flag);
}
#endif

fan::vec2 loco_t::get_mouse_position(const camera_t& camera, const viewport_t& viewport) const {
  return fan::graphics::screen_to_world(get_raw_mouse_position(), viewport, camera);
}

fan::vec2 loco_t::get_mouse_position(const fan::graphics::render_view_t& render_view) const {
  return get_mouse_position(render_view.camera, render_view.viewport);
}

fan::vec2 loco_t::get_raw_mouse_position() const {
  return window.get_mouse_position();
}

fan::vec2 loco_t::translate_position(const fan::vec2& p, viewport_t viewport, camera_t camera) {
  auto v = viewport_get(viewport);
  fan::vec2 d = v.size;

  auto c = camera_get(camera);
  f32_t l = c.coordinates.left;
  f32_t r = c.coordinates.right;
  f32_t t = c.coordinates.top;
  f32_t b = c.coordinates.bottom;

  fan::vec2 tp = (p - v.position) / d;
  tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
  return tp;
}

fan::vec2 loco_t::translate_position(const fan::vec2& p) {
  return translate_position(p, orthographic_render_view.viewport, orthographic_render_view.camera);
}

// shared key state helper - avoids 6 near-identical functions
bool loco_t::key_state_is(int key, bool include_repeat) {
  int s = window.key_state(key);
  if (include_repeat) {
    return s == (int)fan::mouse_state::press || s == (int)fan::mouse_state::repeat;
  }
  return s == (int)fan::mouse_state::press;
}

bool loco_t::is_mouse_clicked(int button) { return key_state_is(button, false); }
bool loco_t::is_mouse_down(int button) { return key_state_is(button, true); }
bool loco_t::is_mouse_released(int button) {
  return window.key_state(button) == (int)fan::mouse_state::release;
}

fan::vec2 loco_t::get_mouse_drag(int button) {
  if (is_mouse_down(button)) {
    if (window.drag_delta_start != fan::vec2(-1)) {
      return window.get_mouse_position() - window.drag_delta_start;
    }
  }
  return fan::vec2();
}

bool loco_t::is_key_clicked(int key) { return key_state_is(key, false); }
bool loco_t::is_key_down(int key) { return key_state_is(key, true); }
bool loco_t::is_key_released(int key) {
  return window.key_state(key) == (int)fan::mouse_state::release;
}

bool loco_t::is_active(std::string_view action_name, int pstate) {
  return input.input_action.is_active(action_name, pstate);
}

bool loco_t::is_toggled(std::string_view action_name) {
  return input.input_action.is_toggled(action_name);
}

bool loco_t::is_toggled(int key) {
  return input.input_action.is_toggled(key);
}

bool loco_t::is_toggled(std::initializer_list<int> keys) {
  return input.input_action.is_toggled(keys);
}

bool loco_t::is_clicked(std::string_view action_name) {
  return input.input_action.is_clicked(action_name);
}

bool loco_t::is_down(std::string_view action_name) {
  return input.input_action.is_down(action_name);
}

bool loco_t::is_released(std::string_view action_name) {
  return input.input_action.is_released(action_name);
}

#if defined(FAN_2D)
void loco_t::shape_open(
  std::uint16_t shape_type,
  std::size_t sizeof_vi,
  std::size_t sizeof_ri,
  fan::graphics::shader_t shader,
  fan::graphics::shaper_t::ShapeRenderDataSize_t instance_count,
  bool instanced,
  std::uint8_t draw_mode
) {
  fan::graphics::shaper_t::BlockProperties_t bp;
  bp.MaxElementPerBlock = (fan::graphics::shaper_t::MaxElementPerBlock_t)fan::graphics::MaxElementPerBlock;
  bp.RenderDataSize = (decltype(fan::graphics::shaper_t::BlockProperties_t::RenderDataSize))(sizeof_vi * instance_count);
  bp.DataSize = sizeof_ri;
  std::construct_at(&bp.renderer.vk);
  fan::graphics::shaper_t::BlockProperties_t::vk_t vk;

  // 2 for rect instance, upv
  static constexpr auto vulkan_buffer_count = 3;
  decltype(vk.shape_data.m_descriptor)::properties_t rectp;
  auto& shaderd = *(fan::vulkan::shader_t*)gloco()->context_functions.shader_get(&gloco()->context.vk, shader);
  std::uint32_t ds_offset = 2;
  vk.shape_data.open(gloco()->context.vk, 1);
  vk.shape_data.allocate(gloco()->context.vk, std::max<std::uint64_t>(bp.RenderDataSize * bp.MaxElementPerBlock, 16));
  std::array<fan::vulkan::write_descriptor_set_t, vulkan_buffer_count> ds_properties {{{0}}};
  {
    ds_properties[0].binding = 0;
    ds_properties[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ds_properties[0].flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    ds_properties[0].range = VK_WHOLE_SIZE;
    ds_properties[0].buffer = vk.shape_data.common.memory[gloco()->get_context().vk.current_frame].buffer;
    ds_properties[0].dst_binding = 0;

    ds_properties[1].binding = 1;
    ds_properties[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ds_properties[1].flags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    ds_properties[1].buffer = shaderd.projection_view_block->common.memory[gloco()->get_context().vk.current_frame].buffer;
    ds_properties[1].range = shaderd.projection_view_block->m_size;
    ds_properties[1].dst_binding = 1;

    VkDescriptorImageInfo imageInfo {};
    auto img = gloco()->image_get(gloco()->default_texture).vk;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.imageView = img.image_view;
    imageInfo.sampler = img.sampler;

    ds_properties[2].use_image = 1;
    ds_properties[2].binding = 2;
    ds_properties[2].dst_binding = 2;
    ds_properties[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ds_properties[2].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (std::uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
      ds_properties[ds_offset].image_infos[i] = imageInfo;
    }
  }

  vk.shape_data.open_descriptors(gloco()->context.vk, {ds_properties.begin(), ds_properties.end()});
  vk.shape_data.m_descriptor.update(context.vk, 3, 0);
  fan::vulkan::context_t::pipeline_t p;
  fan::vulkan::context_t::pipeline_t::properties_t pipe_p{};
  VkPipelineColorBlendAttachmentState attachment = fan::vulkan::get_default_color_blend();
  if (shape_type == fan::graphics::shapes::shape_type_t::light) {
    attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  }
  pipe_p.color_blend_attachments = {attachment};
  pipe_p.shader = shader;
  pipe_p.descriptor_layouts = {vk.shape_data.m_descriptor.m_layout};
  pipe_p.push_constants_size = sizeof(fan::vulkan::context_t::push_constants_t);
  pipe_p.enable_depth_test = false;
  pipe_p.shape_type = (VkPrimitiveTopology)fan::graphics::get_draw_mode(draw_mode);
  p.open(context.vk, pipe_p);
  vk.pipeline = p;
  bp.renderer.vk = std::move(vk);

  fan::graphics::g_shapes->shaper.SetShapeType(shape_type, bp);
}
#endif

fan::graphics::shader_t loco_t::get_sprite_shader(const std::string_view fragment_file_path, const std::string& fragment) {
  return fan::graphics::get_sprite_shader(fragment_file_path, fragment);
}

std::string loco_t::get_renderer_string() {  return "Vulkan";
}

std::string_view loco_t::get_platform_string() {
#if defined(fan_platform_windows)
  return "Windows";
#elif defined(fan_platform_linux)
  return "Linux";
#else
  return "Unknown";
#endif
}

std::string_view loco_t::get_build_string() {
#if defined(FAN_DEBUG_BUILD)
  return "Release";
#else
  return "Debug";
#endif
}

std::string_view loco_t::get_physics_string() {
#if defined(FAN_PHYSICS_2D)
  return "Box2D";
#else
  return "disabled";
#endif
}

#if defined(FAN_GUI)

void loco_t::toggle_console() {
  gui.render_console = !gui.render_console;
}

void loco_t::toggle_console(bool active) {
  gui.render_console = active;
}

#endif

fan::graphics::image_load_properties_t loco_t::default_noise_image_properties() {
  fan::graphics::image_load_properties_t lp;
  lp.format = fan::graphics::image_format_e::rgb_unorm;
  lp.internal_format = fan::graphics::image_format_e::rgb_unorm;
  lp.min_filter = fan::graphics::image_filter_e::linear;
  lp.mag_filter = fan::graphics::image_filter_e::linear;
  lp.visual_output = fan::graphics::image_sampler_address_mode_e::mirrored_repeat;
  return lp;
}

fan::graphics::image_t loco_t::create_noise_image(const fan::vec2& size) {
  return create_noise_image(size, fan::random::value_i64(0, ((std::uint32_t)-1) / 2));
}

fan::graphics::image_t loco_t::create_noise_image(const fan::vec2& size, int seed) {
  fan::noise_t noise(seed);
  auto data = noise.generate_data(size);
  auto lp = default_noise_image_properties();
  fan::image::info_t ii {(void*)data.data(), size, 3};
  return image_load(ii, lp);
}

fan::graphics::image_t loco_t::create_noise_image(const fan::vec2& size, const std::vector<std::uint8_t>& data) {
  auto lp = default_noise_image_properties();
  fan::image::info_t ii {(void*)data.data(), size, 3};
  return image_load(ii, lp);
}

fan::vec2 loco_t::convert_mouse_to_ndc(const fan::vec2& mouse_position) const {
  return fan::math::convert_position_ndc(mouse_position, window.get_size());
}

fan::vec2 loco_t::convert_mouse_to_ndc() const {
  return fan::math::convert_position_ndc(get_mouse_position(), window.get_size());
}

fan::ray3_t loco_t::convert_mouse_to_ray(const fan::vec3& camera_position, const fan::mat4& projection, const fan::mat4& view) {
  return fan::math::convert_position_to_ray(get_mouse_position(), window.get_size(), camera_position, projection, view);
}

fan::ray3_t loco_t::convert_mouse_to_ray(const fan::mat4& projection, const fan::mat4& view) {
  return fan::math::convert_position_to_ray(get_mouse_position(), window.get_size(), camera_get_position(perspective_render_view.camera), projection, view);
}

void loco_t::set_clear_color(const fan::color& color) {
#if defined(FAN_GUI)
  get_clear_color() = get_smenu(this)->config.post_processing.clear_color = color;
#else
  get_clear_color() = color;
#endif
}

#if defined(loco_cuda)
void loco_t::cuda_textures_t::close(loco_t* loco, fan::graphics::shapes::shape_t& cid) {
  loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)cid.GetData(fan::graphics::g_shapes->shaper);
  std::uint8_t image_amount = fan::graphics::get_channel_amount(ri.format);
  for (std::uint32_t i = 0; i < image_amount; ++i) {
    wresources[i].close();
    if (ri.images_rest[i] != loco->default_texture) {
      gloco()->image_unload(ri.images_rest[i]);
    }
    ri.images_rest[i] = loco->default_texture;
  }
  inited = false;
}

void loco_t::cuda_textures_t::resize(loco_t* loco, fan::graphics::shapes::shape_t& id, std::uint8_t format, fan::vec2ui size) {
  auto vi_image = id.get_image();
  if (vi_image.iic() || vi_image == loco->default_texture) {
    id.reload(format, size);
  }
  auto& ri = *(universal_image_renderer_t::ri_t*)id.GetData(loco->shaper);
  if (inited == false) {
    id.reload(format, size);
    vi_image = id.get_image();
    std::uint8_t image_amount = fan::graphics::get_channel_amount(format);
    for (std::uint32_t i = 0; i < image_amount; ++i) {
      if (i == 0) {
        wresources[i].open(gloco()->image_get_handle(vi_image));
      }
      else {
        wresources[i].open(gloco()->image_get_handle(ri.images_rest[i - 1]));
      }
    }
    inited = true;
  }
  else {
    if (gloco()->image_get_data(vi_image).size == size) {
      return;
    }
    for (std::uint32_t i = 0; i < fan::graphics::get_channel_amount(ri.format); ++i) {
      wresources[i].close();
    }
    id.reload(format, size);
    vi_image = id.get_image();
    ri = *(universal_image_renderer_t::ri_t*)id.GetData(loco->shaper);
    std::uint8_t image_amount = fan::graphics::get_channel_amount(format);
    for (std::uint32_t i = 0; i < image_amount; ++i) {
      if (i == 0) {
        wresources[i].open(gloco()->image_get_handle(vi_image));
      }
      else {
        wresources[i].open(gloco()->image_get_handle(ri.images_rest[i - 1]));
      }
    }
  }
}

loco_t::cudaArray_t& loco_t::cuda_textures_t::get_array(std::uint32_t index_t) {
  return wresources[index_t].cuda_array;
}

void loco_t::cuda_textures_t::graphics_resource_t::open(int texture_id) {
  fan::cuda::check_error(cudaGraphicsGLRegisterImage(&resource, texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));
  map();
}

void loco_t::cuda_textures_t::graphics_resource_t::close() {
  if (resource == nullptr) {
    return;
  }
  unmap();
  fan::cuda::check_error(cudaGraphicsUnregisterResource(resource));
  resource = nullptr;
}

void loco_t::cuda_textures_t::graphics_resource_t::map() {
  fan::cuda::check_error(cudaGraphicsMapResources(1, &resource, 0));
  fan::cuda::check_error(cudaGraphicsSubResourceGetMappedArray(&cuda_array, resource, 0, 0));
}

void loco_t::cuda_textures_t::graphics_resource_t::unmap() {
  fan::cuda::check_error(cudaGraphicsUnmapResources(1, &resource));
}
#endif


#if defined(FAN_2D)

void loco_t::camera_move_to(const fan::graphics::shapes::shape_t& shape, const fan::graphics::render_view_t& render_view) {
  camera_set_center(orthographic_render_view.camera, shape.get_position());
}

void loco_t::camera_move_to(const fan::graphics::shapes::shape_t& shape) {
  camera_move_to(shape, orthographic_render_view);
}

void loco_t::camera_move_to_smooth(const fan::graphics::shapes::shape_t& shape, const fan::graphics::render_view_t& render_view) {
  fan::vec2 current = camera_get_position(render_view.camera);
  fan::vec2 target = shape.get_position();
  camera_set_center(orthographic_render_view.camera, current.lerp(target, f32_t(1.f - std::exp(-15.f * get_delta_time()))));
}

void loco_t::camera_move_to_smooth(const fan::graphics::shapes::shape_t& shape) {
  camera_move_to_smooth(shape, orthographic_render_view);
}

bool loco_t::shader_update_fragment(std::uint16_t shape_type, const std::string_view fragment_file_path, const std::string& fragment) {
  auto shader_nr = shader_get_nr(shape_type);
  auto& shader_data = shader_get_data(shape_type);
  shader_set_vertex(shader_nr, shader_data.path_vertex, shader_data.svertex);
  shader_set_fragment(shader_nr, fragment_file_path, fragment);
  return shader_compile(shader_nr);
}
#endif

#if defined(FAN_GUI)
namespace fan::graphics::gui {
  void process_frame() {
    auto it = gloco()->gui.gui_draw_cb.GetNodeFirst();
    while (it != gloco()->gui.gui_draw_cb.dst) {
      gloco()->gui.gui_draw_cb.StartSafeNext(it);
      gloco()->gui.gui_draw_cb[it]();
      it = gloco()->gui.gui_draw_cb.EndSafeNext();
    }
  }

  void render_allocations_plot() {
    using namespace fan::graphics;

#if defined(fan_std23)
    bool hovered = false;
#endif

    struct pause_state_t {
      bool paused;
    };

    static pause_state_t pause_state = {false};

    ::gui::checkbox("pause updates", &pause_state.paused);

    static std::vector<f32_t> allocation_sizes;
    static std::vector<fan::memory::heap_profiler_t::memory_data_t> allocations;
    static f32_t max_y = 0;

    if (!pause_state.paused) {
      allocation_sizes.clear();
      allocations.clear();
      max_y = 0;

      struct alloc_view_t {
        void* p;
        std::size_t n;
      };

      static std::vector<alloc_view_t> sorted_allocs;

      auto& profiler = fan::memory::heap_profiler_t::instance();

      sorted_allocs.clear();
      sorted_allocs.reserve(profiler.get_memory_map().size());
      allocations.reserve(profiler.get_memory_map().size());

      {
        bool was_enabled = profiler.enabled;
        profiler.enabled = false;

        std::lock_guard<std::mutex> lock(profiler.memory_mutex);

        for (auto const& kv : profiler.get_memory_map()) {
          sorted_allocs.push_back(alloc_view_t {kv.first, kv.second.n});
          allocations.push_back(kv.second);
        }

        profiler.enabled = was_enabled;
      }

      std::sort(sorted_allocs.begin(), sorted_allocs.end(),
        [](auto const& a, auto const& b) {
        return (std::uintptr_t)a.p < (std::uintptr_t)b.p;
      });

      for (auto const& av : sorted_allocs) {
        f32_t v = (f32_t)av.n / (1024 * 1024);
        allocation_sizes.push_back(v);
        max_y = max_y < v ? v : max_y;
      }
    }

    auto& profiler = fan::memory::heap_profiler_t::instance();
    ::gui::text("Active allocations:", profiler.memory_map.size());
    ::gui::text("Allocation size:", profiler.current_allocation_size / 1e6, " (MB)");

    int total_mem_MB, used_MB;
    gloco()->get_vram_usage(&total_mem_MB, &used_MB);

    if (used_MB != -1) { ::gui::text("VRAM used memory", used_MB, " (MB)"); }
    if (total_mem_MB != -1) { ::gui::text("VRAM total memory", total_mem_MB, " (MB)"); }

    VmaTotalStatistics stats;
    vmaCalculateStatistics(gloco()->context.vk.allocator, &stats);
    ::gui::text("VMA block overhead", (stats.total.statistics.blockBytes - stats.total.statistics.allocationBytes) / (1024 * 1024), " (MB)");
    ::gui::text("VMA total block memory", stats.total.statistics.blockBytes / (1024 * 1024), " (MB)");
    ::gui::text("VMA total alloc memory", stats.total.statistics.allocationBytes / (1024 * 1024), " (MB)");
    ::gui::text("VMA block count", stats.total.statistics.blockCount);
    ::gui::text("VMA alloc count", stats.total.statistics.allocationCount);

    std::uint64_t tracked_ram_mb = profiler.current_allocation_size / 1e6;
    std::uint64_t tracked_vram_mb = 0;

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(gloco()->context.vk.physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
      if (stats.memoryType[i].statistics.blockCount > 0) {
        VkMemoryPropertyFlags flags = mem_props.memoryTypes[i].propertyFlags;
        std::string flags_str = "";
        if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) flags_str += "DL ";
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) flags_str += "HV ";
        if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) flags_str += "HC ";
        if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) flags_str += "CA ";

        std::uint64_t block_mb = stats.memoryType[i].statistics.blockBytes / (1024 * 1024);
        if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
          tracked_vram_mb += block_mb;
        } else {
          tracked_ram_mb += block_mb;
        }

        std::string text = std::string("Type ") + std::to_string(i) + " [" + flags_str + "]" +
          " | Blocks: " + std::to_string(stats.memoryType[i].statistics.blockCount) +
          " | Allocs: " + std::to_string(stats.memoryType[i].statistics.allocationCount) +
          " | Block: " + std::to_string(block_mb) + "MB" +
          " | Alloc: " + std::to_string(stats.memoryType[i].statistics.allocationBytes / (1024 * 1024)) + "MB";
        ::gui::text(text);
      }
    }

    if (gloco()->context.vk.swap_chain_images.size()) {
      VkMemoryRequirements req;
      vkGetImageMemoryRequirements(gloco()->context.vk.device, gloco()->context.vk.swap_chain_images[0], &req);
      std::uint64_t swapchain_mb = (req.size * gloco()->context.vk.swap_chain_images.size()) / (1024 * 1024);
      ::gui::text("Swapchain memory", swapchain_mb, " (MB)");
      tracked_vram_mb += swapchain_mb;
    }

    if (gloco()->context.vk.vai_depth.size() && gloco()->context.vk.vai_depth[0].image != VK_NULL_HANDLE) {
      VkMemoryRequirements req_depth;
      vkGetImageMemoryRequirements(gloco()->context.vk.device, gloco()->context.vk.vai_depth[0].image, &req_depth);
      std::uint64_t depth_mb = (req_depth.size * gloco()->context.vk.vai_depth.size()) / (1024 * 1024);
      ::gui::text("Depth memory", depth_mb, " (MB)");
      tracked_vram_mb += depth_mb;
    }

    ::gui::text("---------------------------------");
    ::gui::text("Tracked RAM", tracked_ram_mb, " (MB)");
    ::gui::text("Tracked VRAM", tracked_vram_mb, " (MB)");

    fan::vec2 cursor_pos = ::gui::get_cursor_pos();
    fan::vec2 window_size = ::gui::get_window_size();
    fan::vec2 available_size = window_size - cursor_pos;

  #if defined(fan_std23)
    static std::stacktrace stack;
  #endif

    if (allocation_sizes.size() &&
      ::gui::plot::begin_plot("Memory Allocations", available_size,
        ::gui::plot::flags_no_frame | ::gui::plot::flags_no_legend)) {
      ::gui::plot::setup_axis(::gui::plot::axis_y1, "Memory (MB)");
      ::gui::plot::setup_axis_limits(::gui::plot::axis_y1, 0, max_y);
      ::gui::plot::setup_axis(::gui::plot::axis_x1, "Allocations");
      ::gui::plot::setup_axis_limits(::gui::plot::axis_x1, 0, (double)allocation_sizes.size());

      ::gui::plot::push_style_var(::gui::plot::style_var_fill_alpha, 0.25f);
      ::gui::plot::plot_bars("Allocations", allocation_sizes.data(), allocation_sizes.size());
      ::gui::plot::pop_style_var();

      if (::gui::plot::is_plot_hovered()) {
        auto mouse = ::gui::plot::get_plot_mouse_pos();
        mouse.x = (int)mouse.x;

        f32_t half_width = 0.25f;
        f32_t tool_l = ::gui::plot::plot_to_pixels(mouse.x - half_width * 1.5f, mouse.y).x;
        f32_t tool_r = ::gui::plot::plot_to_pixels(mouse.x + half_width * 1.5f, mouse.y).x;
        f32_t tool_t = ::gui::plot::get_plot_pos().y;
        f32_t tool_b = tool_t + ::gui::plot::get_plot_size().y;

        ::gui::plot::push_plot_clip_rect();
        auto draw_list = ::gui::get_window_draw_list();
        draw_list->AddRectFilled(fan::vec2(tool_l, tool_t), fan::vec2(tool_r, tool_b),
          fan::color(128, 128, 128, 64).get_gui_color());
        ::gui::plot::pop_plot_clip_rect();

      #if defined(fan_std23)
        if (mouse.x >= 0 && mouse.x < allocation_sizes.size()) {
          if (fan::window::is_mouse_clicked()) {
            pause_state.paused = true;
            open_popup("view stack");
          }
          stack = allocations[(int)mouse.x].line_data;
          hovered = true;
        }
      #endif
      }

    #if defined(fan_std23)
      if (hovered) {
        begin_tooltip();
        std::ostringstream oss;
        oss << stack;
        std::string stack_str = oss.str();
        std::string final_str;
        std::size_t pos = 0;

        for (;;) {
          auto end = stack_str.find(')', pos);
          if (end == std::string::npos) break;
          end += 1;
          auto begin = stack_str.rfind('\\', end);
          if (begin == std::string::npos) break;
          begin += 1;
          final_str += stack_str.substr(begin, end - begin);
          final_str += "\n";
          pos = end + 1;
        }

        text(final_str);
        end_tooltip();
      }

      if (begin_popup("view stack", ::gui::window_flags_always_horizontal_scrollbar)) {
        std::ostringstream oss;
        oss << stack;
        text(oss.str());
        end_popup();
      }
    #endif

      ::gui::plot::end_plot();
    }
  }
} // namespace fan::graphics::gui
#endif

void shader_set_camera(fan::graphics::shader_t nr, fan::graphics::camera_t camera_nr) {
  fan::graphics::get_vk_context().shaders.shader_set_camera(nr, camera_nr);
}

loco_t::properties_t fan::get_centered_window(vec2 size) {
  return {
    .window_position = (vec2(get_primary_screen_resolution()) - size) * 0.5f,
    .window_size = size
  };
}

void fan::stage_loader_t::nr_t::erase() {
  #if FAN_DEBUG >= 2
  if (iic()) {
    fan::throw_error("double erase or uninitialized erase");
  }
  #endif
  gstage->close_stage(*this);
  sic();
}

fan::graphics::async_image_t loco_t::image_load_async(
  const std::string& path,
  const fan::graphics::image_load_properties_t& properties
) {
  fan::graphics::async_image_t out;
  out.image = default_texture;
  out.result = fan::image::async_cache().load(path);

  async_image_uploads.push_back({
    .image = out.image,
    .properties = properties,
    .result = out.result
  });

  return out;
}

void loco_t::async_image_process() {
  std::size_t uploaded = 0;

  for (std::size_t i = 0; i < async_image_uploads.size();) {
    auto& u = async_image_uploads[i];

    if (!u.result->try_finish()) {
      ++i;
      continue;
    }

    if (u.result->state == fan::image::async_result_t::state_e::ready) {
      if (uploaded >= max_async_image_uploads_per_frame) {
        ++i;
        continue;
      }

      fan::image::info_t info;
      info.data = u.result->image.data.get();
      info.size = u.result->image.size;
      info.channels = u.result->image.channels;

      fan::graphics::image_reload(u.image, info, u.properties);
      ++uploaded;
    }

    async_image_uploads[i] = std::move(async_image_uploads.back());
    async_image_uploads.pop_back();
  }
}

fan::event::task_t fan::stage_loader_t::change_stage_impl(
  std::function<void()> close_cb,
  std::function<void()> open_cb,
  fan::stage_fade_mode_t mode,
  f32_t duration,
  fan::color color
) {
#if defined(FAN_2D)
  if (mode == fan::stage_fade_mode_t::instant || duration <= 0.f) {
    close_cb();
    open_cb();
    co_return;
  }
  fan::graphics::shape_t overlay{fan::graphics::shapes::rectangle_t::properties_t{
    .position = fan::vec3(0, 0, 0xfffe),
    .size = fan::vec2(99999),
    .color = color.set_alpha(mode == fan::stage_fade_mode_t::fade_in ? 1.f : 0.f),
    .blending = true,
  }};
  f32_t half = (mode == fan::stage_fade_mode_t::crossfade) ? duration / 2.f : duration;
  if (mode == fan::stage_fade_mode_t::crossfade) {
    for (f32_t t = 0.f; t < half; t += gloco()->get_delta_time()) {
      overlay.set_color(color.set_alpha(t / half));
      co_await fan::graphics::co_next_frame();
    }
  }
  close_cb();
  open_cb();
  for (f32_t t = 0.f; t < half; t += gloco()->get_delta_time()) {
    overlay.set_color(color.set_alpha(1.f - t / half));
    co_await fan::graphics::co_next_frame();
  }
#else
  co_return;
#endif
}

void fan::stage_loader_t::close_stage(fan::stage_loader_t::nr_t id) {
  auto* sc = (stage_common_t*)stage_list[id].stage;
  auto update_nr = stage_list[id].update_nr;
  gloco()->m_update_callback.unlrec(update_nr);
  sc->close(stage_list[id].stage);
  stage_list.unlrec(id);
}

#endif

#endif