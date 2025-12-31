module;

#define loco_framebuffer
#define loco_post_process

#include <fan/utility.h>

// TODO REMOVE
#include <fan/graphics/opengl/init.h>
#if defined(FAN_VULKAN)
  // TODO REMOVE
#include <vulkan/vulkan.h>
#endif

#include <uv.h>
#undef min
#undef max

#include <source_location>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <utility>
#include <coroutine>
#include <iostream>

#if defined(fan_std23)
  #include <stacktrace>
#endif

#if defined(FAN_GUI)
  #include <iomanip>
#endif

module fan.graphics.loco;

#if defined(FAN_JSON)
namespace fan {
  std::pair<size_t, size_t> json_stream_parser_t::find_next_json_bounds(std::string_view s, size_t pos) const noexcept {
    pos = s.find('{', pos);
    if (pos == std::string::npos) return {pos, pos};

    int depth = 0;
    bool in_str = false;

    for (size_t i = pos; i < s.length(); i++) {
      char c = s[i];
      if (c == '"' && (i == 0 || s[i - 1] != '\\')) in_str = !in_str;
      else if (!in_str) {
        if (c == '{') depth++;
        else if (c == '}' && --depth == 0) return {pos, i + 1};
      }
    }
    return {pos, std::string::npos};
  }

  std::vector<json_stream_parser_t::parsed_result> json_stream_parser_t::process(std::string_view chunk) {
    std::vector<parsed_result> results;
    buf += chunk;
    size_t pos = 0;

    while (pos < buf.length()) {
      auto [start, end] = find_next_json_bounds(buf, pos);
      if (start == std::string::npos) break;
      if (end == std::string::npos) {
        buf = buf.substr(start);
        break;
      }

      try {
        results.push_back({true, fan::json::parse(buf.data() + start, buf.data() + end - start), ""});
      }
      catch (const fan::json::parse_error& e) {
        results.push_back({false, fan::json{}, e.what()});
      }

      pos = buf.find('{', end);
      if (pos == std::string::npos) pos = end;
    }

    buf = pos < buf.length() ? buf.substr(pos) : "";
    return results;
  }

  void json_stream_parser_t::clear() noexcept { buf.clear(); }
}
#endif

#if defined(FAN_GUI)
namespace fan {
  namespace graphics {
    namespace gui {
      void render_allocations_plot();
      void process_frame();
    }
  }
}
#endif

namespace fan::graphics {

  std::uint32_t get_draw_mode(std::uint8_t internal_draw_mode) {
    if (gloco()->get_renderer() == fan::window_t::renderer_t::opengl) {
    #if defined(FAN_OPENGL)
      return fan::opengl::core::get_draw_mode(internal_draw_mode);
    #endif
    }
    else if (gloco()->get_renderer() == fan::window_t::renderer_t::vulkan) {
    #if defined(FAN_VULKAN)
      return fan::vulkan::core::get_draw_mode(internal_draw_mode);
    #endif
    }
  #if FAN_DEBUG >= fan_debug_medium
    fan::throw_error("invalid get");
  #endif
    return -1;
  }
}

uint8_t loco_t::get_renderer() {
  return window.renderer;
}

fan::graphics::shader_nr_t loco_t::shader_create() {
  return context_functions.shader_create(&context);
}

fan::graphics::context_shader_t loco_t::shader_get(fan::graphics::shader_nr_t nr) {
  fan::graphics::context_shader_t context_shader;
  if (window.renderer == fan::window_t::renderer_t::opengl) {
    context_shader.gl = *(fan::opengl::context_t::shader_t*)context_functions.shader_get(&context, nr);
  }
#if defined(FAN_VULKAN)
  else if (window.renderer == fan::window_t::renderer_t::vulkan) {
    context_shader.vk = *(fan::vulkan::context_t::shader_t*)context_functions.shader_get(&context, nr);
  }
#endif 
  return context_shader;
}

void loco_t::shader_erase(fan::graphics::shader_nr_t nr) {
  context_functions.shader_erase(&context, nr);
}

void loco_t::shader_use(fan::graphics::shader_nr_t nr) {
  context_functions.shader_use(&context, nr);
}

void loco_t::shader_set_vertex(fan::graphics::shader_nr_t nr, const std::string& vertex_code) {
  context_functions.shader_set_vertex(&context, nr, vertex_code);
}

void loco_t::shader_set_fragment(fan::graphics::shader_nr_t nr, const std::string& fragment_code) {
  context_functions.shader_set_fragment(&context, nr, fragment_code);
}

bool loco_t::shader_compile(fan::graphics::shader_nr_t nr) {
  return context_functions.shader_compile(&context, nr);
}

void loco_t::shader_set_camera(shader_t nr, camera_t camera_nr) {
  if (window.renderer == fan::window_t::renderer_t::opengl) {
    context.gl.shader_set_camera(nr, camera_nr);
  }
#if defined(FAN_VULKAN)
  else if (window.renderer == fan::window_t::renderer_t::vulkan) {
    fan::throw_error("todo");
  }
#endif 
}

#if defined(FAN_2D)
  fan::graphics::shader_nr_t loco_t::shader_get_nr(uint16_t shape_type) {
    return fan::graphics::g_shapes->shaper.GetShader(shape_type);
  }

  fan::graphics::shader_list_t::nd_t& loco_t::shader_get_data(uint16_t shape_type) {
    return loco_t::shader_list[shader_get_nr(shape_type)];
  }
#endif

std::vector<uint8_t> loco_t::image_get_pixel_data(fan::graphics::image_nr_t nr, int image_format, fan::vec2 uvp, fan::vec2 uvs) {
  if (window.renderer == fan::window_t::renderer_t::opengl) {
    return context_functions.image_get_pixel_data(&context, nr, fan::opengl::context_t::global_to_opengl_format(image_format), uvp, uvs);
  }
  else {
    fan::throw_error("");
    return {};
  }
}

fan::graphics::image_nr_t loco_t::image_create() {
  return context_functions.image_create(&context);
}

fan::graphics::context_image_t loco_t::image_get(fan::graphics::image_nr_t nr) {
  fan::graphics::context_image_t img;
  if (window.renderer == fan::window_t::renderer_t::opengl) {
    img.gl = *(fan::opengl::context_t::image_t*)context_functions.image_get(&context, nr);
  }
#if defined(FAN_VULKAN)
  else if (window.renderer == fan::window_t::renderer_t::vulkan) {
    img.vk = *(fan::vulkan::context_t::image_t*)context_functions.image_get(&context, nr);
  }
#endif 
  return img;
}

uint64_t loco_t::image_get_handle(fan::graphics::image_nr_t nr) {
  return context_functions.image_get_handle(&context, nr);
}

fan::graphics::image_data_t& loco_t::image_get_data(fan::graphics::image_nr_t nr) {
  return image_list[nr];
}

void loco_t::image_erase(fan::graphics::image_nr_t nr) {
  context_functions.image_erase(&context, nr);
}

void loco_t::image_bind(fan::graphics::image_nr_t nr) {
  context_functions.image_bind(&context, nr);
}

void loco_t::image_unbind(fan::graphics::image_nr_t nr) {
  context_functions.image_unbind(&context, nr);
}

fan::graphics::image_load_properties_t& loco_t::image_get_settings(fan::graphics::image_nr_t nr) {
  return context_functions.image_get_settings(&context, nr);
}

void loco_t::image_set_settings(fan::graphics::image_nr_t nr, const fan::graphics::image_load_properties_t& settings) {
  context_functions.image_set_settings(&context, nr, settings);
}

fan::graphics::image_nr_t loco_t::image_load(const fan::image::info_t& image_info) {
  return context_functions.image_load_info(&context, image_info);
}

fan::graphics::image_nr_t loco_t::image_load(const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
  return context_functions.image_load_info_props(&context, image_info, p);
}

fan::graphics::image_nr_t loco_t::image_load(const std::string& path, const std::source_location& callers_path) {
  return context_functions.image_load_path(&context, path, callers_path);
}

fan::graphics::image_nr_t loco_t::image_load(const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path) {
  return context_functions.image_load_path_props(&context, path, p, callers_path);
}

fan::graphics::image_nr_t loco_t::image_load(fan::color* colors, const fan::vec2ui& size) {
  return context_functions.image_load_colors(&context, colors, size);
}

fan::graphics::image_nr_t loco_t::image_load(fan::color* colors, const fan::vec2ui& size, const fan::graphics::image_load_properties_t& p) {
  return context_functions.image_load_colors_props(&context, colors, size, p);
}

void loco_t::image_unload(fan::graphics::image_nr_t nr) {
  context_functions.image_unload(&context, nr);
}

bool loco_t::is_image_valid(fan::graphics::image_nr_t nr) {
  return nr != default_texture && nr.iic() == false;
}

fan::graphics::image_nr_t loco_t::create_missing_texture() {
  return context_functions.create_missing_texture(&context);
}

fan::graphics::image_nr_t loco_t::create_transparent_texture() {
  return context_functions.create_transparent_texture(&context);
}

void loco_t::image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info) {
  context_functions.image_reload_image_info(&context, nr, image_info);
}

void loco_t::image_reload(fan::graphics::image_nr_t nr, const fan::image::info_t& image_info, const fan::graphics::image_load_properties_t& p) {
  context_functions.image_reload_image_info_props(&context, nr, image_info, p);
}

void loco_t::image_reload(fan::graphics::image_nr_t nr, const std::string& path, const std::source_location& callers_path) {
  context_functions.image_reload_path(&context, nr, path, callers_path);
}

void loco_t::image_reload(fan::graphics::image_nr_t nr, const std::string& path, const fan::graphics::image_load_properties_t& p, const std::source_location& callers_path) {
  context_functions.image_reload_path_props(&context, nr, path, p, callers_path);
}

fan::graphics::image_nr_t loco_t::image_create(const fan::color& color) {
  return context_functions.image_create_color(&context, color);
}

fan::graphics::image_nr_t loco_t::image_create(const fan::color& color, const fan::graphics::image_load_properties_t& p) {
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

void loco_t::camera_set_target(fan::graphics::camera_nr_t nr, const fan::vec2& target, f32_t move_speed) {
  fan::vec2 src = camera_get_position(orthographic_render_view.camera);
  camera_set_position(
    orthographic_render_view.camera,
    move_speed == 0 ? target : src + (target - src) * delta_time * move_speed
  );
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
  fan::vec2 position = viewport_get_position(nr);
  context_functions.viewport_set_nr(&context, nr, position, viewport_size, window.get_size());
}

void loco_t::viewport_set_position(fan::graphics::viewport_nr_t nr, const fan::vec2& viewport_position) {
  fan::vec2 size = viewport_get_size(nr);
  context_functions.viewport_set_nr(&context, nr, viewport_position, size, window.get_size());
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

void loco_t::camera_move(fan::graphics::context_camera_t& camera, f64_t dt, f32_t movement_speed, f32_t friction) {
  camera.velocity /= friction * dt + 1;
  static constexpr auto minimum_velocity = 0.001;
  static constexpr f32_t camera_rotate_speed = 100;
  if (camera.velocity.x < minimum_velocity && camera.velocity.x > -minimum_velocity) {
    camera.velocity.x = 0;
  }
  if (camera.velocity.y < minimum_velocity && camera.velocity.y > -minimum_velocity) {
    camera.velocity.y = 0;
  }
  if (camera.velocity.z < minimum_velocity && camera.velocity.z > -minimum_velocity) {
    camera.velocity.z = 0;
  }

  f64_t msd = (movement_speed * dt);
  if (window.key_pressed(fan::input::key_w)) {
    camera.velocity += camera.m_front * msd;
  }
  if (window.key_pressed(fan::input::key_s)) {
    camera.velocity -= camera.m_front * msd;
  }
  if (window.key_pressed(fan::input::key_a)) {
    camera.velocity -= camera.m_right * msd;
  }
  if (window.key_pressed(fan::input::key_d)) {
    camera.velocity += camera.m_right * msd;
  }

  if (window.key_pressed(fan::input::key_space)) {
    camera.velocity.y += msd;
  }
  if (window.key_pressed(fan::input::key_left_shift)) {
    camera.velocity.y -= msd;
  }

  f64_t rotate = camera.sensitivity * camera_rotate_speed * delta_time;
  if (window.key_pressed(fan::input::key_left)) {
    camera.set_yaw(camera.get_yaw() - rotate);
  }
  if (window.key_pressed(fan::input::key_right)) {
    camera.set_yaw(camera.get_yaw() + rotate);
  }
  if (window.key_pressed(fan::input::key_up)) {
    camera.set_pitch(camera.get_pitch() + rotate);
  }
  if (window.key_pressed(fan::input::key_down)) {
    camera.set_pitch(camera.get_pitch() - rotate);
  }

  camera.position += camera.velocity * delta_time;
  camera.update_view();

  camera.m_view = camera.get_view_matrix();
}

#if defined(FAN_2D)

void loco_t::add_shape_to_immediate_draw(fan::graphics::shapes::shape_t&& s) {
  immediate_render_list.emplace_back(std::move(s));
}

uint32_t loco_t::add_shape_to_static_draw(fan::graphics::shapes::shape_t&& s) {
  uint32_t ret = s.NRI;
  static_render_list[ret] = std::move(s);
  return ret;
}

void loco_t::remove_static_shape_draw(const fan::graphics::shapes::shape_t& s) {
  static_render_list.erase(s.NRI);
}
#endif


void loco_t::generate_commands(loco_t* loco) {
#if defined(FAN_GUI)
  loco->console.open();

  loco->console.commands.add("echo", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    fan::commands_t::output_t out;
    out.text = fan::append_args(args) + "\n";
    out.highlight = fan::graphics::highlight_e::info;
    loco->console.commands.output_cb(out);
  }).description = "prints something - usage echo [args]";

  loco->console.commands.add("help", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.empty()) {
      fan::commands_t::output_t out;
      out.highlight = fan::graphics::highlight_e::info;
      std::string out_str;
      out_str += "{\n";
      for (const auto& i : loco->console.commands.func_table) {
        out_str += "\t" + i.first + ",\n";
      }
      out_str += "}\n";
      out.text = out_str;
      loco->console.commands.output_cb(out);
      return;
    }
    else if (args.size() == 1) {
      auto found = loco->console.commands.func_table.find(args[0]);
      if (found == loco->console.commands.func_table.end()) {
        loco->console.commands.print_command_not_found(args[0]);
        return;
      }
      fan::commands_t::output_t out;
      out.text = found->second.description + "\n";
      out.highlight = fan::graphics::highlight_e::info;
      loco->console.commands.output_cb(out);
    }
    else {
      loco->console.commands.print_invalid_arg_count();
    }
  }).description = "get info about specific command - usage help command";

  loco->console.commands.add("list", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    std::string out_str;
    for (const auto& i : loco->console.commands.func_table) {
      out_str += i.first + "\n";
    }

    fan::commands_t::output_t out;
    out.text = out_str;
    out.highlight = fan::graphics::highlight_e::info;

    loco->console.commands.output_cb(out);
  }).description = "lists all commands - usage list";

  loco->console.commands.add("alias", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() < 2 || args[1].empty()) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    if (loco->console.commands.insert_to_command_chain(args)) {
      return;
    }
    loco->console.commands.func_table[args[0]] = loco->console.commands.func_table[args[1]];
  }).description = "can create alias commands - usage alias [cmd name] [cmd]";


  loco->console.commands.add("show_fps", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    loco->show_fps = std::stoi(args[0]);
  }).description = "toggles fps - usage show_fps [value]";

  loco->console.commands.add("quit", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    exit(0);
  }).description = "quits program - usage quit";

  loco->console.commands.add("clear", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    loco->console.output_buffer.clear();
    loco->console.editor.SetText("");
  }).description = "clears output buffer - usage clear";

#if defined(loco_framebuffer)
  loco->console.commands.add("set_gamma", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    loco->shader_set_value(loco->gl.m_fbo_final_shader, "gamma", std::stof(args[0]));
  }).description = "sets gamma for postprocessing shader";

  loco->console.commands.add("set_gamma", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    loco->shader_set_value(loco->gl.m_fbo_final_shader, "gamma", std::stof(args[0]));
  }).description = "sets gamma for postprocessing shader";
  loco->console.commands.add("set_contrast", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    loco->shader_set_value(loco->gl.m_fbo_final_shader, "contrast", std::stof(args[0]));
  }).description = "sets contrast for postprocessing shader";

  loco->console.commands.add("set_exposure", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    loco->shader_set_value(loco->gl.m_fbo_final_shader, "exposure", std::stof(args[0]));
  }).description = "sets exposure for postprocessing shader";

  loco->console.commands.add("set_bloom_strength", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    loco->settings_menu.bloom_strength = std::stof(args[0]);
    loco->shader_set_value(loco->gl.m_fbo_final_shader, "bloom_strength", loco->settings_menu.bloom_strength);
  }).description = "sets bloom strength for postprocessing shader";
#endif
  loco->console.commands.add("set_vsync", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    loco->set_vsync(std::stoi(args[0]));
  }).description = "sets vsync";

  loco->console.commands.add("set_target_fps", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    loco->set_target_fps(std::stoi(args[0]));
  }).description = "sets target fps";

  loco->console.commands.add("debug_memory", [nr = loco_t::update_callback_handle_t()](fan::console_t* self, const fan::commands_t::arg_t& args) mutable {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    if (nr.iic() && std::stoi(args[0])) {
      nr = loco->add_update_callback([] (void* loco_ptr) {
        fan::graphics::gui::set_next_window_bg_alpha(0.99f);
        static int init = 0;
        fan::graphics::gui::window_flags_t window_flags = fan::graphics::gui::window_flags_no_title_bar | fan::graphics::gui::window_flags_no_focus_on_appearing;
        if (init == 0) {
          fan::graphics::gui::set_next_window_size(fan::vec2(600, 300));
          init = 1;
        }
        fan::graphics::gui::begin("fan_memory_dbg_wnd", nullptr, window_flags);
        fan::graphics::gui::render_allocations_plot();
        fan::graphics::gui::end();
      });
    }
    else if (!nr.iic() && !std::stoi(args[0])) {
      loco->remove_update_callback(nr);
    }
  }).description = "opens memory debug window";

  loco->console.commands.add("set_clear_color", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    loco->clear_color = fan::color::parse(args[0]);
  }).description = "sets clear color of window - input example {1,0,0,1} red";

  loco->console.commands.add("set_lighting_ambient", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }
    loco->lighting.set_target(fan::color::parse(args[0]));
  }).description = "sets clear color of window - input example {1,0,0,1} red";

#if defined(FAN_2D)
  // shapes
  loco->console.commands.add("rectangle", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() < 1 || args.size() > 3) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }

    try {
      fan::graphics::shapes::rectangle_t::properties_t props;
      props.position = fan::vec3::parse(args[0]);
      // optional
      if (args.size() >= 2) props.size = fan::vec2::parse(args[1]);
      // optional
      props.color = args.size() == 3 ? fan::color::parse(args[2]) : fan::colors::white;

      auto NRI = loco->add_shape_to_static_draw(props);
      loco->console.println_colored(
        "Added rectangle",
        fan::colors::green
      );
      loco->console.println(
        "  id: " + std::to_string(NRI) +
        "\n  position " + (std::string)props.position +
        "\n  size " + (std::string)props.size +
        "\n  color " + props.color.to_string(),
        fan::graphics::highlight_e::info
      );
    }
    catch (const std::exception& e) {
      loco->console.println_colored("Invalid arguments: " + std::string(e.what()), fan::colors::red);
    }
  }).description = "Adds static rectangle {x,y[,z]} {w,h} [{r,g,b,a}]";

  loco->console.commands.add("remove_shape", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    if (args.size() != 1) {
      loco->console.commands.print_invalid_arg_count();
      return;
    }

    try {
      uint32_t shape_id = std::stoull(args[0]);
      //shape_id
      fan::graphics::shapes::shape_t* s = reinterpret_cast<fan::graphics::shapes::shape_t*>(&shape_id);
      loco->remove_static_shape_draw(*s);
      loco->console.println_colored(
        "Removed shape with id {}" + std::to_string(shape_id),
        fan::colors::green
      );
    }
    catch (const std::exception& e) {
      loco->console.println_colored(
        "Invalid argument: " + std::string(e.what()),
        fan::colors::red
      );
    }
  }).description = "Removes a shape by its id";
#endif

  loco->console.commands.add("print", [](fan::console_t* self, const fan::commands_t::arg_t& args) {
    auto* loco = OFFSETLESS(self, loco_t, console);
    fan::commands_t::output_t out;
    out.text = fan::append_args(args) + "\n";
    out.highlight = fan::graphics::highlight_e::info;
    loco->text_logger.print(fan::graphics::highlight_color_table[out.highlight], out.text);
  }).description = "prints something to bottom left of screen - usage print [args]";

#endif
}

#if defined(FAN_2D)

void loco_t::culling_rebuild_grid() {
  fan::graphics::culling::static_grid_init(
    shapes.visibility.static_grid,
    world_min,
    cell_size,
    grid_size
  );

  fan::graphics::culling::dynamic_grid_init(
    shapes.visibility.dynamic_grid,
    world_min,
    cell_size,
    grid_size
  );
}

void loco_t::rebuild_static_culling() {
  fan::graphics::culling::rebuild_static(shapes.visibility);
}

bool loco_t::culling_enabled() const {
  return shapes.visibility.enabled;
}

void loco_t::set_culling_enabled(bool enabled) {
  fan::graphics::culling::set_enabled(shapes.visibility, enabled);
}

void loco_t::get_culling_stats(uint32_t& visible, uint32_t& culled) const {
  visible = 0;
  uint32_t total = 0;
  for (auto const& [cam_id, cam_state] : shapes.visibility.camera_states) {
    visible += std::count(cam_state.visible.begin(), cam_state.visible.end(), 1);
    total = std::max<uint32_t>(total, cam_state.visible.size());
  }
  culled = (total >= visible) ? (total - visible) : 0;
}

void loco_t::run_culling() {
  fan::graphics::camera_list_t::nrtra_t nrtra;
  fan::graphics::camera_nr_t nr;
  nrtra.Open(&camera_list, &nr);

  auto& culling = shapes.visibility;
  while (nrtra.Loop(&camera_list, &nr)) {
    if (nr == perspective_render_view.camera) {
      continue;
    }
    fan::graphics::culling::cull_camera(
      culling,
      fan::graphics::g_shapes->shaper,
      nr
    );
  }

  nrtra.Close(&camera_list);
}

void loco_t::set_cull_padding(const fan::vec2& padding) {
  shapes.visibility.padding = padding;
}

void loco_t::visualize_culling() {
  const auto& cam = camera_get();
  
  fan::vec2 top_left = fan::graphics::screen_to_world(fan::vec2(0, 0), orthographic_render_view);
  fan::vec2 bottom_right = fan::graphics::screen_to_world(viewport_get_size(), orthographic_render_view);
  
  fan::vec2 scaled_padding = shapes.visibility.padding / cam.zoom;
  fan::vec2 padded_min = top_left - scaled_padding;
  fan::vec2 padded_max = bottom_right + scaled_padding;
  
  fan::vec2 top_right(padded_max.x, padded_min.y);
  fan::vec2 bottom_left(padded_min.x, padded_max.y);
  
  add_shape_to_immediate_draw(fan::graphics::shapes::line_t::properties_t{
    .src = padded_min, 
    .dst = top_right,
    .color = fan::color(1, 0, 0, 0.8f),
    .thickness = 5.f / cam.zoom
  });
  add_shape_to_immediate_draw(fan::graphics::shapes::line_t::properties_t{
    .src = top_right, 
    .dst = padded_max,
    .color = fan::color(1, 0, 0, 0.8f),
    .thickness = 5.f / cam.zoom
  });
  add_shape_to_immediate_draw(fan::graphics::shapes::line_t::properties_t{
    .src = padded_max, 
    .dst = bottom_left,
    .color = fan::color(1, 0, 0, 0.8f),
    .thickness = 5.f / cam.zoom
  });
  add_shape_to_immediate_draw(fan::graphics::shapes::line_t::properties_t{
    .src = bottom_left, 
    .dst = padded_min,
    .color = fan::color(1, 0, 0, 0.8f),
    .thickness = 5.f / cam.zoom
  });
}
#endif

#if defined(FAN_VULKAN)
void loco_t::check_vk_result(VkResult err) {
  if (err != VK_SUCCESS) {
    fan::print("vkerr", (int)err);
  }
}
#endif

#if defined(FAN_GUI)

void loco_t::init_gui() {
  if (fan::graphics::gui::g_gui_initialized) {
    gui_initialized = true;
    return;
  }
  fan::graphics::gui::init(
    window,
    window.renderer,
    fan::window_t::renderer_t::opengl,
    fan::window_t::renderer_t::vulkan
  #if defined(FAN_VULKAN)
    ,
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
  #endif
  );
  gui_initialized = true;
}

void loco_t::destroy_gui() {
  if (!gui_initialized || !fan::graphics::gui::g_gui_initialized) {
    return;
  }

  fan::graphics::gui::shutdown_graphics_context(
    window.renderer,
    fan::window_t::renderer_t::opengl,
    fan::window_t::renderer_t::vulkan
  #if defined(FAN_VULKAN)
    , context.vk.device
  #endif
  );
  if (reload_renderer_to != (decltype(reload_renderer_to))-1) {
    gui_initialized = false;
    return;
  }
#if defined(FAN_VULKAN)
  if (window.renderer == fan::window_t::renderer_t::vulkan) {
    context.vk.gui_close(); // TODO remove
  }
#endif

  fan::graphics::gui::destroy();
  gui_initialized = false;
}
#endif

loco_t::loco_t() : loco_t(loco_t::properties_t()) {

}

loco_t::loco_t(const loco_t::properties_t& p) {

  fan::graphics::engine_init_cbs.Open(); // leak, double open

  auto& ctx = fan::graphics::ctx();

  // init globals
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
  ctx.input_action = &input_action;
  ctx.lighting = &lighting;

#if defined(FAN_GUI)
  ctx.console = &console;
  ctx.text_logger = &text_logger;
#endif

#if defined(FAN_2D)
  shapes.texture_pack = &texture_pack;
  shapes.immediate_render_list = &immediate_render_list;
  shapes.static_render_list = &static_render_list;
#endif

  input_action.is_active_func = [this](int key) -> int {
    return window.key_state(key);
  };
#if defined(FAN_2D)

  fan::graphics::shaper_t::gl_add_shape_type() = [](
    fan::graphics::shaper_t::ShapeTypes_NodeData_t& nd,
    const fan::graphics::shaper_t::BlockProperties_t& bp) {
    gloco()->gl.add_shape_type(nd, bp); // dont look here
  };
  fan::graphics::g_shapes = &shapes;
#endif

#if defined(FAN_GUI)
  fan::graphics::gui::profile_heap(
    [](size_t size, void* user_data) -> void* {
    return fan::heap_profiler_t::instance().allocate_memory(size); // malloc
  },
    [](void* ptr, void* user_data) {
    fan::heap_profiler_t::instance().deallocate_memory(ptr); // free
  }
  );
#endif

#if defined(fan_platform_windows)
  // use utf8 for console output
  SetConsoleOutputCP(CP_UTF8);
#endif

  if (fan::init_manager_t::initialized() == false) {
    fan::init_manager_t::initialize();
  }
  render_shapes_top = p.render_shapes_top;
  window.renderer = p.renderer;
  if (window.renderer == fan::window_t::renderer_t::opengl) {
    new (&context.gl) fan::opengl::context_t();
    context_functions = fan::graphics::get_gl_context_functions();
    gl.open();
  }

  window.set_antialiasing(p.samples);
  window.open(p.window_size, fan::window_t::default_window_name, p.window_flags, p.window_open_mode);
  gloco() = this;


#if FAN_DEBUG >= fan_debug_high && !defined(FAN_VULKAN)
  if (window.renderer == fan::window_t::renderer_t::vulkan) {
    fan::throw_error("trying to use vulkan renderer, but FAN_VULKAN build flag is disabled");
  }
#endif

#if defined(FAN_VULKAN)
  if (window.renderer == fan::window_t::renderer_t::vulkan) {
    context_functions = fan::graphics::get_vk_context_functions();
    new (&context.vk) fan::vulkan::context_t();
    context.vk.enable_clear = !render_shapes_top;
    context.vk.shapes_top = render_shapes_top;
    context.vk.open(window);
  }
#endif

  start_time.start();

  //fan::print("less pain", this, (void*)&lighting, (void*)((uint8_t*)&lighting - (uint8_t*)this), sizeof(*this), lighting.ambient);
  if (window.renderer == fan::window_t::renderer_t::opengl) {
    window.make_context_current();

  #if FAN_DEBUG >= fan_debug_high
    get_context().gl.set_error_callback();
  #endif

    if (window.get_antialiasing() > 0) {
      glEnable(GL_MULTISAMPLE);
    }

    gl.initialize_fb_vaos();
  }


  load_engine_images();

#if defined(FAN_2D)
  shapes.shaper.Open();
  {

    // filler
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::light, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::light_end, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::visible, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::depth, sizeof(fan::graphics::depth_t), fan::graphics::shaper_t::KeyBitOrderLow);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::blending, sizeof(fan::graphics::blending_t), fan::graphics::shaper_t::KeyBitOrderLow);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::image, sizeof(fan::graphics::image_t), fan::graphics::shaper_t::KeyBitOrderLow);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::viewport, sizeof(fan::graphics::viewport_t), fan::graphics::shaper_t::KeyBitOrderAny);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::camera, sizeof(loco_t::camera_t), fan::graphics::shaper_t::KeyBitOrderAny);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::ShapeType, sizeof(fan::graphics::shaper_t::ShapeTypeIndex_t), fan::graphics::shaper_t::KeyBitOrderAny);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::filler, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::draw_mode, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::vertex_count, sizeof(uint32_t), fan::graphics::shaper_t::KeyBitOrderAny);
    fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::shadow, sizeof(uint8_t), fan::graphics::shaper_t::KeyBitOrderAny);

    //fan::graphics::g_shapes->shaper.AddKey(fan::graphics::Key_e::image4, sizeof(fan::graphics::image_t) * 4, fan::graphics::shaper_t::KeyBitOrderLow);
  }
  // order of open needs to be same with shapes enum
#endif

  {
    fan::vec2 window_size = window.get_size();
    {
      orthographic_render_view.create_default(window_size);
    }
    {
      perspective_render_view.camera = open_camera_perspective();
      perspective_render_view.viewport = open_viewport(
        fan::vec2(0, 0),
        window_size
      );
    }
  }

  if (window.renderer == fan::window_t::renderer_t::opengl) {
    gl.init();
  }
#if defined(FAN_VULKAN)
  else if (window.renderer == fan::window_t::renderer_t::vulkan) {
    vk.init();
  }
#endif

  #if defined(FAN_2D)
    if (window.renderer == fan::window_t::renderer_t::opengl) {
      gl.shapes_open();
    }
  #if defined(FAN_VULKAN)
    else if (window.renderer == fan::window_t::renderer_t::vulkan) {
      vk.shapes_open();
    }
  #endif
  #endif

#if defined(FAN_GUI)
  init_gui();
  generate_commands(this);

  settings_menu.open();
#endif

  setup_input_callbacks();

  auto it = fan::graphics::engine_init_cbs.GetNodeFirst();
  while (it != fan::graphics::engine_init_cbs.dst) {
    fan::graphics::engine_init_cbs.StartSafeNext(it);
    fan::graphics::engine_init_cbs[it](this);
    it = fan::graphics::engine_init_cbs.EndSafeNext();
  }

#if defined(FAN_AUDIO)

  if (system_audio.Open() != 0) {
    fan::throw_error("failed to open fan audio");
  }
  audio.bind(&system_audio);
  fan::audio::piece_hover.open_piece("audio/hover.sac", 0);
  fan::audio::piece_click.open_piece("audio/click.sac", 0);

  fan::audio::g_audio = &audio;

#endif

  fan::graphics::ctx().default_texture = default_texture;

#if defined(FAN_GUI)
  console.commands.call("debug_memory " + std::to_string((int)fan::heap_profiler_t::instance().enabled));
#endif

#if defined(FAN_2D)
  set_culling_enabled(true);
  cell_size = 256;
  culling_rebuild_grid();
  //shapes.visibility.padding = fan::vec2(1000, 1000);
#endif

  set_vsync(false); // using libuv
}

loco_t::~loco_t() {
  destroy();
}

void loco_t::destroy() {
#if defined(FAN_2D)
  // TODO fix destruct order to not do manually, because shaper closes before them?
  static_render_list.clear();
  immediate_render_list.clear();

#endif

  if (window == nullptr) {
    return;
  }

#if defined(FAN_GUI)
  console.commands.func_table.clear();
  console.close();
#endif

#if defined(FAN_OPENGL)
  if (window.renderer == fan::window_t::renderer_t::opengl) {
    gl.close();
  }
#endif
#if defined(FAN_2D)
  fan::graphics::g_shapes->shaper.Close();
#endif
#if defined(FAN_VULKAN)
  if (window.renderer == fan::window_t::renderer_t::vulkan) {
    vkDeviceWaitIdle(context.vk.device);
    vkDestroySampler(context.vk.device, vk.post_process_sampler, nullptr);
    vk.d_attachments.close(context.vk);
    vk.post_process.close(context.vk);
  }
#endif
#if defined(FAN_GUI)
  destroy_gui();
#endif
  window.close();
#if defined(FAN_AUDIO)
  audio.unbind();
  system_audio.Close();
#endif
}

void loco_t::close() {
  destroy();
}

void loco_t::setup_input_callbacks() {

#if defined(FAN_GUI)
  input_action.add(fan::key_escape, "open_settings");
#endif

  input_action.add({fan::key_a}, "move_left");
  input_action.add({fan::key_d}, "move_right");
  input_action.add({fan::key_w}, "move_forward");
  input_action.add({fan::key_s}, "move_back");
  input_action.add({fan::key_space, fan::key_w, fan::gamepad_a}, "move_up");

#if defined(FAN_PHYSICS_2D)
  input_action.add_keycombo({fan::key_left_control, fan::key_5}, "debug_physics");
#endif

  #if defined(FAN_2D)
  buttons_handle = window.add_buttons_callback([](const fan::window_t::buttons_data_t& d) {
    fan::vec2 pos = fan::vec2(d.window->get_mouse_position());
    fan::graphics::g_shapes->vfi.feed_mouse_button(d.button, d.state);
  });
  #endif

  keys_handle = window.add_keys_callback([&, windowed = true](const fan::window_t::keys_data_t& d) mutable {
  #if defined(FAN_2D)
    fan::graphics::g_shapes->vfi.feed_keyboard(d.key, d.state);
  #endif
    auto* loco = OFFSETLESS(d.window, loco_t, window);
    if (d.key == fan::key_enter && d.state == fan::keyboard_state::press && loco->window.key_pressed(fan::key_left_alt)) {
      windowed = !windowed;
      loco->window.set_display_mode(windowed ? fan::window_t::mode::windowed : fan::window_t::mode::borderless);
    }
  });

#if defined(FAN_2D)
  mouse_move_handle = window.add_mouse_move_callback([&](const fan::window_t::mouse_move_data_t& d) {
    fan::graphics::g_shapes->vfi.feed_mouse_move(d.position);
  });

  text_callback_handle = window.add_text_callback([&](const fan::window_t::text_data_t& d) {
    fan::graphics::g_shapes->vfi.feed_text(d.character);
  });
#endif
}

void loco_t::switch_renderer(uint8_t renderer) {
  std::vector<std::string> image_paths;
  fan::vec2 window_size = window.get_size();
  fan::vec2 window_position = window.get_position();
  uint64_t flags = window.flags;

#if defined(FAN_GUI)
  bool was_imgui_init = gui_initialized;
#endif

  {// close
  #if defined(FAN_VULKAN)
    if (window.renderer == fan::window_t::renderer_t::vulkan) {
      // todo wrap to vk.
      vkDeviceWaitIdle(context.vk.device);
      vkDestroySampler(context.vk.device, vk.post_process_sampler, nullptr);
      vk.d_attachments.close(context.vk);
      vk.post_process.close(context.vk);
    #if defined(FAN_2D)
      for (auto& st : fan::graphics::g_shapes->shaper.ShapeTypes) {
        if (st.sti == (decltype(st.sti))-1) {
          continue;
        }
      #if defined(FAN_VULKAN)
        auto& str = st.renderer.vk;
        str.shape_data.close(context.vk);
        str.pipeline.close(context.vk);
      #endif
        //st.BlockList.Close();
      }
    #endif
      //CLOOOOSEEE POSTPROCESSS IMAGEEES
    }
    else
    #endif
      if (window.renderer == fan::window_t::renderer_t::opengl) {
        glDeleteVertexArrays(1, &gl.fb_vao);
        glDeleteBuffers(1, &gl.fb_vbo);
        context.gl.internal_close();
      }

  #if defined(FAN_GUI)
    if (gui_initialized) {
      fan::graphics::gui::shutdown_graphics_context(
        window.renderer,
        fan::window_t::renderer_t::opengl,
        fan::window_t::renderer_t::vulkan
      #if defined(FAN_VULKAN)
        , context.vk.device
      #endif
      );
      fan::graphics::gui::shutdown_window_context();
      gui_initialized = false;
    }
  #endif

    window.close();
  }

  {// reopen
    window.renderer = reload_renderer_to; // i dont like this {window.renderer = ...}
    if (window.renderer == fan::window_t::renderer_t::opengl) {
      context_functions = fan::graphics::get_gl_context_functions();
      new (&context.gl) fan::opengl::context_t();
      gl.open();
    }

    window.open(window_size, fan::window_t::default_window_name, flags | fan::window_t::flags::hidden);
    window.set_position(window_position);
    window.set_position(window_position);
    glfwShowWindow(window);
    window.flags = flags;

  #if defined(FAN_VULKAN)
    if (window.renderer == fan::window_t::renderer_t::vulkan) {
      new (&context.vk) fan::vulkan::context_t();
      context_functions = fan::graphics::get_vk_context_functions();
      context.vk.open(window);
    }
  #endif
  }

  {// reload
    {
      {
        fan::graphics::camera_list_t::nrtra_t nrtra;
        fan::graphics::camera_nr_t nr;
        nrtra.Open(&camera_list, &nr);
        while (nrtra.Loop(&camera_list, &nr)) {
          auto& cam = camera_list[nr];
          camera_set_ortho(
            nr,
            fan::vec2(cam.coordinates.left, cam.coordinates.right),
            fan::vec2(cam.coordinates.top, cam.coordinates.bottom)
          );
        }
        nrtra.Close(&camera_list);
      }
      {
        fan::graphics::viewport_list_t::nrtra_t nrtra;
        fan::graphics::viewport_nr_t nr;
        nrtra.Open(&viewport_list, &nr);
        while (nrtra.Loop(&viewport_list, &nr)) {
          auto& viewport = viewport_list[nr];
          viewport_set(
            nr,
            viewport.position,
            viewport.size
          );
        }
        nrtra.Close(&viewport_list);
      }
    }

    {
      {
        {
          fan::graphics::image_list_t::nrtra_t nrtra;
          fan::graphics::image_nr_t nr;
          nrtra.Open(&image_list, &nr);
          while (nrtra.Loop(&image_list, &nr)) {

            if (window.renderer == fan::window_t::renderer_t::opengl) {
              // illegal
              image_list[nr].internal = new fan::opengl::context_t::image_t;
              fan_opengl_call(glGenTextures(1, &((fan::opengl::context_t::image_t*)context_functions.image_get(&context.gl, nr))->texture_id));
            }
          #if defined(FAN_VULKAN)
            else if (window.renderer == fan::window_t::renderer_t::vulkan) {
              // illegal
              image_list[nr].internal = new fan::vulkan::context_t::image_t;
            }
          #endif
            // handle blur?
            auto image_path = image_list[nr].image_path;
            if (image_path.empty()) {
              fan::image::info_t info;
              info.data = (void*)fan::image::missing_texture_pixels;
              info.size = 2;
              info.channels = 4;
              fan::graphics::image_load_properties_t lp;
              lp.min_filter = fan::graphics::image_filter::nearest;
              lp.mag_filter = fan::graphics::image_filter::nearest;
              lp.visual_output = fan::graphics::image_sampler_address_mode::repeat;
              image_reload(nr, info, lp);
            }
            else {
              image_reload(nr, image_list[nr].image_path);
            }
          }
          nrtra.Close(&image_list);
        }
        {
          fan::graphics::shader_list_t::nrtra_t nrtra;
          fan::graphics::shader_nr_t nr;
          nrtra.Open(&shader_list, &nr);
          while (nrtra.Loop(&shader_list, &nr)) {
            if (window.renderer == fan::window_t::renderer_t::opengl) {
              shader_list[nr].internal = new fan::opengl::context_t::shader_t;
            }
          #if defined(FAN_VULKAN)
            else if (window.renderer == fan::window_t::renderer_t::vulkan) {
              shader_list[nr].internal = new fan::vulkan::context_t::shader_t;
              ((fan::vulkan::context_t::shader_t*)shader_list[nr].internal)->projection_view_block = new std::remove_pointer_t<decltype(fan::vulkan::context_t::shader_t::projection_view_block)>;
            }
          #endif
          }
          nrtra.Close(&shader_list);
        }
      }
      fan::image::info_t info;
      info.data = (void*)fan::image::missing_texture_pixels;
      info.size = 2;
      info.channels = 4;
      fan::graphics::image_load_properties_t lp;
      lp.min_filter = fan::graphics::image_filter::nearest;
      lp.mag_filter = fan::graphics::image_filter::nearest;
      lp.visual_output = fan::graphics::image_sampler_address_mode::repeat;
      image_reload(default_texture, info, lp);
    }

    if (window.renderer == fan::window_t::renderer_t::opengl) {
    #if defined(FAN_2D)
      gl.shapes_open();
    #endif
      gl.initialize_fb_vaos();
      if (window.get_antialiasing() > 0) {
        glEnable(GL_MULTISAMPLE);
      }
    }
  #if defined(FAN_VULKAN)
    else if (window.renderer == fan::window_t::renderer_t::vulkan) {
    #if defined(FAN_2D)
      vk.shapes_open();
    #endif
    }
  #endif

  #if defined(FAN_GUI)
    if (was_imgui_init && fan::graphics::gui::g_gui_initialized) {
      fan::graphics::gui::init_graphics_context(
        window,
        window.renderer,
        fan::window_t::renderer_t::opengl,
        fan::window_t::renderer_t::vulkan
      #if defined(FAN_VULKAN)
        ,
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
      #endif
      );
      gui_initialized = true;
      settings_menu.set_settings_theme();
    }
  #endif

  #if defined(FAN_2D)
    fan::graphics::g_shapes->shaper._BlockListCapacityChange(fan::graphics::shapes::shape_type_t::rectangle, 0, 1);
    fan::graphics::g_shapes->shaper._BlockListCapacityChange(fan::graphics::shapes::shape_type_t::sprite, 0, 1);
  #endif
  #if defined(FAN_AUDIO)
    if (system_audio.Open() != 0) {
      fan::throw_error("failed to open fan audio");
    }
    audio.bind(&system_audio);
  #endif
  }
  reload_renderer_to = -1;
}

void loco_t::shapes_draw() {
  shape_draw_timer.start();
  if (window.renderer == fan::window_t::renderer_t::opengl) {
    gl.shapes_draw();
  }
#if defined(FAN_VULKAN)
  else
    if (window.renderer == fan::window_t::renderer_t::vulkan) {
      vk.shapes_draw();
    }
#endif
  shape_draw_time_s = shape_draw_timer.seconds();

#if defined(FAN_2D)
  immediate_render_list.clear();
#endif
}

void loco_t::process_shapes() {

#if defined(FAN_VULKAN)
  if (window.renderer == fan::window_t::renderer_t::vulkan) {
    if (render_shapes_top == true) {
      vk.begin_render_pass();
    }
  }
#endif
  for (const auto& i : m_pre_draw) {
    i();
  }

  shapes_draw();

  for (const auto& i : m_post_draw) {
    i();
  }

#if defined(FAN_VULKAN)
  if (window.renderer == fan::window_t::renderer_t::vulkan) {
    auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
    if (vk.image_error == VK_SUCCESS) {
      vkCmdNextSubpass(cmd_buffer, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.post_process);
      vkCmdBindDescriptorSets(
        cmd_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        vk.post_process.m_layout,
        0,
        1,
        vk.d_attachments.m_descriptor_set,
        0,
        nullptr
      );

      context.vk.viewport_set(0, window.get_size(), window.get_size());

      VkRect2D sc{}; sc.offset = {0, 0}; sc.extent = { (uint32_t)window.get_size().x, (uint32_t)window.get_size().y}; 
      vkCmdSetScissor(cmd_buffer, 0, 1, &sc);

      // render post process
      vkCmdDraw(cmd_buffer, 6, 1, 0, 0);
    }
    if (render_shapes_top == true) {
      vkCmdEndRenderPass(cmd_buffer);
    }
  }
#endif
}
void loco_t::process_gui() {
  using namespace fan::graphics;
  gui_draw_timer.start();
#if defined(FAN_GUI)
  fan::graphics::gui::process_frame();

  // append
  gui::begin("##global_renderer");
  text_logger.render();
  gui::end();

  if (fan::window::is_key_pressed(fan::key_f3)) {
    render_console = !render_console;

    // force focus xd
    console.input.InsertText("a");
    console.input.SetText("");
    console.init_focus = true;
    console.input.IsFocused() = false;
  }
  if (render_console) {
    console.render();
  }
  if (input_action.is_active("open_settings")) {
    if (render_console) {
      render_console = false;
    }
    else {
      render_settings_menu = !render_settings_menu;
    }
  }
  if (render_settings_menu) {
    settings_menu.render();
  }

  if (show_fps) {
    using namespace fan::graphics;

    gui::window_flags_t window_flags =
      gui::window_flags_no_title_bar |
      gui::window_flags_no_focus_on_appearing;

    gui::set_next_window_bg_alpha(0.99f);
    gui::set_next_window_size(fan::vec2(831.0000, 693.0000), gui::cond_once);
    gui::begin("Performance window", nullptr, window_flags);

    frame_monitor.update(delta_time);
    shape_monitor.update(shape_draw_time_s);
    gui_monitor.update(gui_draw_time_s);

    auto frame_stats = frame_monitor.stats();
    auto shape_stats = shape_monitor.stats();
    auto gui_stats = gui_monitor.stats();

    static auto format_val = [](double v, int prec = 4) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(prec) << v;
      return oss.str();
    };

    gui::text("Current FPS:", std::to_string(static_cast<int>(1.f / delta_time)));
    gui::text("Average FPS:", std::to_string(static_cast<int>(frame_stats.avg_fps())));
    gui::text("Lowest FPS:", std::to_string(static_cast<int>(frame_stats.min_fps())));
    gui::text("Highest FPS:", std::to_string(static_cast<int>(frame_stats.max_fps())));

    gui::text("Frame Time Avg:", format_val(frame_stats.avg_frame_time_s * 1e3) + " ms");
    gui::text("Shape Draw Avg:", format_val(shape_stats.avg_frame_time_s * 1e3) + " ms");
    gui::text("GUI Draw Avg:", format_val(gui_stats.avg_frame_time_s * 1e3) + " ms");



    if (gui::button(frame_monitor.paused ? "Continue" : "Pause")) {
      frame_monitor.paused = !frame_monitor.paused;
      shape_monitor.paused = frame_monitor.paused;
      gui_monitor.paused = frame_monitor.paused;
    }

    if (gui::button("Reset data")) {
      frame_monitor.reset();
      shape_monitor.reset();
      gui_monitor.reset();
    }

    if (gui::plot::begin_plot("Times", fan::vec2(-1, 0), gui::plot::flags_no_frame)) {
      gui::plot::setup_axes("Frame Index", "Frame Time (ms)",
        gui::plot::axis_flags_auto_fit,
        gui::plot::axis_flags_auto_fit | gui::plot::axis_flags_range_fit
      );
      gui::plot::setup_axis_ticks(gui::plot::axis_y1, 0.0, 10.0, 11);
      frame_monitor.plot(this, "Frame Draw Time");
      shape_monitor.plot(this, "Shape Draw Time");
      gui_monitor.plot(this, "GUI Draw Time");

      if (frame_monitor.buffer.size() > time_plot_scroll.view_size) {
        int max_offset = static_cast<int>(frame_monitor.buffer.size()) - time_plot_scroll.view_size;
        gui::slider("Scroll", &time_plot_scroll.scroll_offset, 0, max_offset);
      }
      gui::plot::end_plot();
    }

    gui::text("Frame Draw Time: ", format_val(delta_time * 1e3) + " ms");
    gui::text("Shape Draw Time: ", format_val(shape_draw_time_s * 1e3) + " ms");
    gui::text("GUI Draw Time: ", format_val(gui_draw_time_s * 1e3) + " ms");

    gui::end();
  }

#if defined(loco_framebuffer)

#endif

  fan::graphics::gui::render(
    window.renderer,
    fan::window_t::renderer_t::opengl,
    fan::window_t::renderer_t::vulkan,
    render_shapes_top
  #if defined(FAN_VULKAN)
    ,
    &fan::graphics::get_vk_context(),
    clear_color,
    vk.image_error,
    context.vk.command_buffers[context.vk.current_frame],
    fan::vulkan::context_t::ImGuiFrameRender
  #endif
  );
#endif
#if defined(FAN_GUI)
  fan::graphics::gui::set_want_io();
#endif
  gui_draw_time_s = gui_draw_timer.seconds();
}

void loco_t::get_vram_usage(int* total_mem_MB, int* used_MB) {
  if (glewIsSupported("GL_NVX_gpu_memory_info")) {
    glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, (GLint*)total_mem_MB);

    GLint currently_available_kb = 0;
    glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &currently_available_kb);

    *used_MB = (*total_mem_MB - currently_available_kb) / 1024.0;
    *total_mem_MB = *total_mem_MB / 1024.0;
  }
  else {
    *total_mem_MB = -1;
    *used_MB = -1;
  }
}

void loco_t::time_monitor_t::update(f32_t v) {
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

void loco_t::time_monitor_t::reset() {
  buffer.clear();
  sum = 0.0f;
  min_q.clear();
  max_q.clear();
}

loco_t::time_monitor_t::stats_t loco_t::time_monitor_t::stats() const {
  if (buffer.empty()) return {0,0,0};

  f32_t avg = sum / buffer.size();
  f32_t min = buffer[min_q.front()];
  f32_t max = buffer[max_q.front()];

  return {avg, min, max};
}

#if defined(FAN_GUI)
void loco_t::time_monitor_t::plot(loco_t* loco, const char* label) {
  using namespace fan::graphics;
  if (buffer.empty()) return;

  int plot_count = std::min(loco->time_plot_scroll.view_size, static_cast<int>(buffer.size()));
  static std::vector<f32_t> plot_data;
  plot_data.resize(plot_count);

  if (!paused) {
    int max_start = std::max(0, static_cast<int>(buffer.size()) - loco->time_plot_scroll.view_size);
    loco->time_plot_scroll.scroll_offset = max_start;
  }

  int max_start = std::max(0, static_cast<int>(buffer.size()) - loco->time_plot_scroll.view_size);
  int start = std::min(loco->time_plot_scroll.scroll_offset, max_start);

  for (int i = 0; i < plot_count; ++i) {
    plot_data[i] = buffer[start + i] * 1e3f; // ms
  }

  gui::plot::plot_line(label, plot_data.data(), plot_count);
}
#endif

//struct frame_validator_t {
//  std::unordered_map<uint32_t, fan::graphics::image_t> expected_images;
//  uint64_t frame = 0;
//
//  void snapshot_shapes() {
//    expected_images.clear();
//    fan::graphics::shaper_t::KeyTraverse_t kt;
//    kt.Init(fan::graphics::g_shapes->shaper);
//
//    while (kt.Loop(fan::graphics::g_shapes->shaper)) {
//      if (kt.kti(fan::graphics::g_shapes->shaper) == fan::graphics::Key_e::image) {
//        auto img = *(fan::graphics::image_t*)kt.kd();
//        auto bmid = kt.bmid();
//        expected_images[bmid.gint()] = img;
//      }
//    }
//  }
//
//  void validate_during_draw() {
//    fan::graphics::shaper_t::KeyTraverse_t kt;
//    kt.Init(fan::graphics::g_shapes->shaper);
//
//    while (kt.Loop(fan::graphics::g_shapes->shaper)) {
//      if (kt.kti(fan::graphics::g_shapes->shaper) == fan::graphics::Key_e::image) {
//        auto img = *(fan::graphics::image_t*)kt.kd();
//        auto bmid = kt.bmid();
//
//        if (expected_images.count(bmid.gint()) && 
//          expected_images[bmid.gint()] != img) {
//          fan::print("FRAME", frame, "GLITCH DETECTED!");
//          fan::print("  BMID:", bmid.gint());
//          fan::print("  Expected:", expected_images[bmid.gint()].NRI);
//          fan::print("  Got:", img.NRI);
//
//        #if defined(fan_std23)
//          fan::print("Stack trace:");
//          fan::print(std::stacktrace::current());
//        #endif
//        }
//      }
//    }
//  }
//} frame_validator;

void loco_t::process_render() {
  //frame_validator.snapshot_shapes();
#if defined(FAN_2D)
  if (init_culling) {
    rebuild_static_culling();
    init_culling = false;
  }
#endif


  if (window.renderer == fan::window_t::renderer_t::opengl) {
    gl.begin_process_frame();
  }

  fan::event::deferred_resume_t::process_resumes();

  {
    auto it = m_update_callback.GetNodeFirst();
    while (it != m_update_callback.dst) {
      m_update_callback.StartSafeNext(it);
      auto prev = m_update_callback.SafeNext.NRI;
      m_update_callback[it](this);
      it = m_update_callback.EndSafeNext();
    }
  }

  for (const auto& i : single_queue) {
    i();
  }

  single_queue.clear();

  std::vector<std::coroutine_handle<>> current_frame;
  // swap with pending to 
  std::swap(current_frame, next_frame_awaiter::pending);

  for (const auto& h : current_frame) {
    h.resume();
  }

#if defined(FAN_2D)
  if (is_visualizing_culling) {
    visualize_culling();
  }
#endif

#if defined(FAN_GUI)
  fan::graphics::gui::end();
#endif

  if (window.renderer == fan::window_t::renderer_t::opengl) {
    run_culling();
  }

#if defined(FAN_2D)
  fan::graphics::g_shapes->shaper.ProcessBlockEditQueue();
#endif

#if defined(FAN_VULKAN)
  if (window.renderer == fan::window_t::renderer_t::vulkan) {
    vk.begin_draw();
  }
#endif

  viewport_set(0, window.get_size());

  //frame_validator.validate_during_draw();

  if (render_shapes_top == false) {
    process_shapes();
    process_gui();
  }
  else {
    process_gui();
    process_shapes();
  }
  for (auto& i : draw_end_cb) {
    i();
  }
  if (window.renderer == fan::window_t::renderer_t::opengl) {
    window.swap_buffers();
  }
#if defined(FAN_VULKAN)
  else if (window.renderer == fan::window_t::renderer_t::vulkan) {
  #if !defined(FAN_GUI)
    auto& cmd_buffer = context.vk.command_buffers[context.vk.current_frame];
    // did draw
    vkCmdNextSubpass(cmd_buffer, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdEndRenderPass(cmd_buffer);
  #endif
    if (vk.image_error != VK_SUCCESS) { 
      context.vk.command_buffer_in_use = false; 
    }
    else {
      VkResult err = context.vk.end_render();
      context.vk.recreate_swap_chain(&window, err);
    }
  }
#endif
  //frame_validator.frame++;
}

bool loco_t::should_close() {
  if (window == nullptr) {
    return true;
  }
  return glfwWindowShouldClose(window);
}

bool loco_t::process_frame(const std::function<void()>& cb) {
  window.handle_events();
  time = start_time.seconds();

  if (should_close()) {
    return 1;
  }

  delta_time = window.m_delta_time;

#if defined(FAN_PHYSICS_2D)
  physics_context.begin_frame(delta_time);
#endif

#if defined(FAN_GUI)
  if (reload_renderer_to != (decltype(reload_renderer_to))-1) {
    switch_renderer(reload_renderer_to);
  }

  lighting.update(delta_time);

  fan::graphics::gui::new_frame(
    window.renderer,
    fan::window_t::renderer_t::opengl,
    fan::window_t::renderer_t::vulkan
  );

  using namespace fan::graphics;

  auto& style = gui::get_style();

  gui::push_style_color(gui::col_window_bg, fan::color(0, 0, 0, 0));
  gui::push_style_color(gui::col_docking_empty_bg, fan::color(0, 0, 0, 0));
  gui::dock_space_over_viewport(0, gui::get_main_viewport());

  if (allow_docking || is_key_down(fan::key_left_control)) {
    gui::get_io().ConfigFlags |= gui::config_flags_docking_enable;
  }
  else {
    gui::get_io().ConfigFlags &= ~gui::config_flags_docking_enable;
  }

  gui::pop_style_color(2);
  gui::set_next_window_pos(fan::vec2(0, 0));
  gui::set_next_window_size(fan::vec2(window.get_size()));

  int flags = gui::window_flags_no_docking | gui::window_flags_no_saved_settings |
    gui::window_flags_no_focus_on_appearing | gui::window_flags_no_move |
    gui::window_flags_no_collapse | gui::window_flags_no_background |
    gui::window_flags_no_resize | gui::dock_node_flags_no_docking_split |
    gui::window_flags_no_title_bar | gui::window_flags_no_bring_to_front_on_focus | gui::window_flags_no_inputs;

  if (!enable_overlay) {
    flags |= gui::window_flags_no_nav;
  }

  gui::begin("##global_renderer", nullptr, flags);

  {
    static f32_t font_size = 15.f;
    std::string fps_text = std::to_string(int(1.0 / delta_time));

    gui::push_font(gui::get_font(font_size));

    fan::vec2 box_size = fan::vec2(34, gui::get_font_size() * 1.4f);
    fan::vec2 text_size = gui::calc_text_size(fps_text);

    fan::vec2 fps_pos = fan::vec2(
      window.get_size().x - box_size.x,
      0
    );

    fan::vec2 bg_min = fps_pos;
    fan::vec2 bg_max = fps_pos + box_size;

    gui::get_window_draw_list()->AddRectFilled(
      bg_min,
      bg_max,
      fan::color(0, 0, 0, 1.0f).get_gui_color(),
      0.0f
    );

    fan::vec2 text_pos = fan::vec2(
      fps_pos.x + box_size.x - text_size.x,
      fps_pos.y + (box_size.y - text_size.y) * 0.5f
    );

    gui::get_window_draw_list()->AddText(
      text_pos,
      fan::color(0.0f, 1, 0.0f, 1.f).get_gui_color(),
      fps_text.c_str()
    );

    gui::pop_font();
  }
#endif

  cb();

#if defined(FAN_2D)
  if (force_line_draw) {
    gl.draw_all_shape_aabbs();
  }
#endif

  // user can terminate from main loop
  if (should_close()) {
    return 1;
  }

  process_render();

  return 0;
}

void loco_t::loop(const std::function<void()>& cb) {
  main_loop = cb;
g_loop:
  double delay = std::round(1.0 / target_fps * 1000.0);

  if (!timer_init) {
    uv_timer_init(fan::event::get_loop(), &timer_handle);
    timer_init = true;
  }
  if (!idle_init) {
    uv_idle_init(fan::event::get_loop(), &idle_handle);
    idle_init = true;
  }

  timer_handle.data = this;
  idle_handle.data = this;

  if (target_fps > 0) {
    start_timer();
  }
  else {
    start_idle();
  }

  uv_run(fan::event::get_loop(), UV_RUN_DEFAULT);
  if (should_close() == false) {
    goto g_loop;
  }
}

loco_t::camera_t loco_t::open_camera(const fan::vec2& x, const fan::vec2& y) {
  loco_t::camera_t camera = camera_create();
  camera_set_ortho(camera, fan::vec2(x.x, x.y), fan::vec2(y.x, y.y));
  return camera;
}

loco_t::camera_t loco_t::open_camera_perspective(f32_t fov) {
  loco_t::camera_t camera = camera_create();
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
  const std::string& forward, const std::string& back,
  const std::string& left, const std::string& right
) {
  fan::vec2 v(
    input_action.is_action_down(right) - input_action.is_action_down(left),
    input_action.is_action_down(back) - input_action.is_action_down(forward)
  );
  fan::vec2 v2 = window.get_gamepad_axis(fan::gamepad_left_thumb);
  if (v2) {
    return v2.length() > 0 ? v2.normalized() : v2;
  }
  return v.length() > 0 ? v.normalized() : v;
}

fan::vec2 loco_t::transform_matrix(const fan::vec2& position) {
  fan::vec2 window_size = window.get_size();
  // not custom ortho friendly - made for -1 1
  return position / window_size * 2 - 1;
}

fan::vec2 loco_t::screen_to_ndc(const fan::vec2& screen_pos) {
  fan::vec2 window_size = window.get_size();
  return screen_pos / window_size * 2 - 1;
}

fan::vec2 loco_t::ndc_to_screen(const fan::vec2& ndc_position) {
  fan::vec2 window_size = window.get_size();
  fan::vec2 normalized_position = (ndc_position + 1) / 2;
  return normalized_position * window_size;
}

void loco_t::set_vsync(bool flag) {
  vsync = flag;
  // vulkan vsync is enabled by presentation mode in swap chain
  if (window.renderer == fan::window_t::renderer_t::opengl) {
    context.gl.set_vsync(&window, flag);
  }
#if defined(FAN_VULKAN)
  if (window.renderer == fan::window_t::renderer_t::vulkan) {
    context.vk.set_vsync(&window, flag);
  }
#endif
}

void loco_t::start_timer(){
  double delay;
  if (target_fps <= 0){
    delay = 0;
  }
  else{
    delay = std::floor(1.0 / target_fps * 1000.0 * 0.9);
    if (delay < 1) delay = 1;
  }

  if (delay > 0){
    uv_timer_start(&timer_handle, [](uv_timer_t* handle){
      loco_t* loco = static_cast<loco_t*>(handle->data);

      f64_t elapsed = loco->frame_timer.seconds();
      loco->frame_timer.restart();
      loco->accumulated_time += elapsed;

      if (loco->accumulated_time >= loco->target_frame_time){
        loco->accumulated_time -= loco->target_frame_time;

        if (loco->process_frame(loco->main_loop)){
          uv_timer_stop(handle);
          uv_stop(fan::event::get_loop());
        }
      }
    }, 0, delay);
  }
}

void loco_t::idle_cb(uv_idle_t* handle) {
  loco_t* loco = static_cast<loco_t*>(handle->data);
  if (loco->process_frame(loco->main_loop)) {
    uv_idle_stop(handle);
    uv_stop(fan::event::get_loop());
  }
}

void loco_t::start_idle(bool start_idle) {
  if (!start_idle) {
    return;
  }
  uv_idle_start(&idle_handle, idle_cb);
}

void loco_t::update_timer_interval(bool idle){
  double delay;
  if (target_fps <= 0){
    delay = 0;
  }
  else{
    delay = std::floor(1.0 / target_fps * 1000.0 * 0.9);
    if (delay < 1) delay = 1;
    target_frame_time = 1.0 / target_fps;
  }

  if (delay > 0){
    if (idle_init){
      uv_idle_stop(&idle_handle);
    }

    if (!timer_enabled){
      frame_timer.start();
      accumulated_time = 0.0;
      start_timer();
      timer_enabled = true;
    }
    uv_timer_set_repeat(&timer_handle, delay);
    uv_timer_again(&timer_handle);
  }
  else{
    if (timer_init){
      uv_timer_stop(&timer_handle);
      timer_enabled = false;
    }

    if (idle_init && idle){
      uv_idle_start(&idle_handle, idle_cb);
    }
  }
}

void loco_t::set_target_fps(int32_t new_target_fps, bool idle){
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

void loco_t::remove_update_callback(update_callback_handle_t handle) {
  m_update_callback.unlrec(handle);
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

f64_t loco_t::current_time() const {
  return delta_time;
}

#if defined(FAN_PHYSICS_2D)
void loco_t::update_physics() {
  physics_context.step(delta_time);
}
#endif

fan::vec2 loco_t::get_mouse_position(const camera_t& camera, const viewport_t& viewport) const {
  return fan::graphics::screen_to_world(get_mouse_position(), viewport, camera);
}

fan::vec2 loco_t::get_mouse_position(const fan::graphics::render_view_t& render_view) const {
  return get_mouse_position(render_view.camera, render_view.viewport);
}

fan::vec2 loco_t::get_mouse_position() const {
  return window.get_mouse_position();
  //return get_mouse_position(gloco()->default_camera->camera, gloco()->default_camera->viewport); behaving oddly
}

fan::vec2 loco_t::translate_position(const fan::vec2& p, viewport_t viewport, camera_t camera) {
  auto v = viewport_get(viewport);
  fan::vec2 viewport_position = v.position;
  fan::vec2 viewport_size = v.size;

  auto c = camera_get(camera);

  f32_t l = c.coordinates.left;
  f32_t r = c.coordinates.right;
  f32_t t = c.coordinates.top;
  f32_t b = c.coordinates.bottom;

  fan::vec2 tp = p - viewport_position;
  fan::vec2 d = viewport_size;
  tp /= d;
  tp = fan::vec2(r * tp.x - l * tp.x + l, b * tp.y - t * tp.y + t);
  return tp;
}

fan::vec2 loco_t::translate_position(const fan::vec2& p) {
  return translate_position(p, orthographic_render_view.viewport, orthographic_render_view.camera);
}

bool loco_t::is_mouse_clicked(int button) {
  return window.key_state(button) == (int)fan::mouse_state::press;
}

bool loco_t::is_mouse_down(int button) {
  int state = window.key_state(button);
  return
    state == (int)fan::mouse_state::press ||
    state == (int)fan::mouse_state::repeat;
}

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

bool loco_t::is_key_pressed(int key) {
  return window.key_state(key) == (int)fan::mouse_state::press;
}

bool loco_t::is_key_down(int key) {
  int state = window.key_state(key);
  return
    state == (int)fan::mouse_state::press ||
    state == (int)fan::mouse_state::repeat;
}

bool loco_t::is_key_released(int key) {
  return window.key_state(key) == (int)fan::mouse_state::release;
}

#if defined(FAN_2D)
void loco_t::shape_open(
  uint16_t shape_type,
  std::size_t sizeof_vi,
  std::size_t sizeof_ri,
  fan::graphics::shape_gl_init_list_t shape_shader_locations,
  const std::string& vertex,
  const std::string& fragment,
  fan::graphics::shaper_t::ShapeRenderDataSize_t instance_count,
  bool instanced
) {
  fan::graphics::shader_t shader = shader_create();

  shader_set_vertex(shader,
    fan::graphics::read_shader(vertex)
  );

  shader_set_fragment(shader,
    fan::graphics::read_shader(fragment)
  );

  shader_compile(shader);

  fan::graphics::shaper_t::BlockProperties_t bp;
  bp.MaxElementPerBlock = (fan::graphics::shaper_t::MaxElementPerBlock_t)fan::graphics::MaxElementPerBlock;
  bp.RenderDataSize = (decltype(fan::graphics::shaper_t::BlockProperties_t::RenderDataSize))(sizeof_vi * instance_count);
  bp.DataSize = sizeof_ri;

  if (window.renderer == fan::window_t::renderer_t::opengl) {
    std::construct_at(&bp.renderer.gl);
    fan::graphics::shaper_t::BlockProperties_t::gl_t d;
    d.locations = shape_shader_locations;
    d.shader = shader;
    d.instanced = instanced;
    bp.renderer.gl = d;
  }
#if defined(FAN_VULKAN)
  else if (window.renderer == fan::window_t::renderer_t::vulkan) {
    std::construct_at(&bp.renderer.vk);
    fan::graphics::shaper_t::BlockProperties_t::vk_t vk;

    // 2 for rect instance, upv
    static constexpr auto vulkan_buffer_count = 3;
    decltype(vk.shape_data.m_descriptor)::properties_t rectp;
    // image
    //uint32_t ds_offset = 3;
    auto& shaderd = *(fan::vulkan::context_t::shader_t*)gloco()->context_functions.shader_get(&gloco()->context.vk, shader);
    uint32_t ds_offset = 2;
    vk.shape_data.open(gloco()->context.vk, 1);
    vk.shape_data.allocate(gloco()->context.vk, 0xffffff);

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
      for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
        ds_properties[ds_offset].image_infos[i] = imageInfo;
      }

      //imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      //imageInfo.imageView = gloco()->get_context().vk.postProcessedColorImageViews[0].image_view;
      //imageInfo.sampler = sampler;

      //imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      //ds_properties[ds_offset + 1].use_image = 1;
      //ds_properties[ds_offset + 1].binding = 4;
      //ds_properties[ds_offset + 1].dst_binding = 4;
      //ds_properties[ds_offset + 1].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
      //ds_properties[ds_offset + 1].flags = VK_SHADER_STAGE_FRAGMENT_BIT;
      //for (uint32_t i = 0; i < fan::vulkan::max_textures; ++i) {
      //  ds_properties[ds_offset + 1].image_infos[i] = imageInfo;
      //}
    }

    vk.shape_data.open_descriptors(gloco()->context.vk, {ds_properties.begin(), ds_properties.end()});
    vk.shape_data.m_descriptor.update(context.vk, 3, 0);
    fan::vulkan::context_t::pipeline_t p;
    fan::vulkan::context_t::pipeline_t::properties_t pipe_p;
    VkPipelineColorBlendAttachmentState attachment = fan::vulkan::get_default_color_blend();
    pipe_p.color_blend_attachment_count = 1;
    pipe_p.color_blend_attachment = &attachment;
    pipe_p.shader = shader;
    pipe_p.descriptor_layout = &vk.shape_data.m_descriptor.m_layout;
    pipe_p.descriptor_layout_count = /*vulkan_buffer_count*/1;
    pipe_p.push_constants_size = sizeof(fan::vulkan::context_t::push_constants_t);
    p.open(context.vk, pipe_p);
    vk.pipeline = p;
    bp.renderer.vk = vk;
  }
#endif

  fan::graphics::g_shapes->shaper.SetShapeType(shape_type, bp);
}
#endif

fan::graphics::shader_t loco_t::get_sprite_vertex_shader(const std::string& fragment) {
  if (get_renderer() == fan::window_t::renderer_t::opengl) {
    fan::graphics::shader_t shader = shader_create();
    shader_set_vertex(
      shader,
      fan::graphics::read_shader("shaders/opengl/2D/objects/sprite.vs")
    );
    shader_set_fragment(shader, fragment);
    if (!shader_compile(shader)) {
      shader_erase(shader);
      shader.sic();
    }
    return shader;
  }
  else {
    fan::print("todo");
  }
  return {};
}

#if defined(FAN_GUI)

void loco_t::toggle_console() {
  render_console = !render_console;
}
void loco_t::toggle_console(bool active) {
  render_console = active;
}

#endif

fan::graphics::image_load_properties_t loco_t::default_noise_image_properties() {
  fan::graphics::image_load_properties_t lp;
  lp.format = fan::graphics::image_format::rgb_unorm;
  lp.internal_format = fan::graphics::image_format::rgb_unorm;
  lp.min_filter = fan::graphics::image_filter::linear;
  lp.mag_filter = fan::graphics::image_filter::linear;
  lp.visual_output = fan::graphics::image_sampler_address_mode::mirrored_repeat;
  return lp;
}

fan::graphics::image_t loco_t::create_noise_image(const fan::vec2& size, int seed) {
  fan::noise_t noise(seed);
  auto data = noise.generate_data(size);
  auto lp = default_noise_image_properties();
  fan::image::info_t ii {(void*)data.data(), size, 3};
  return image_load(ii, lp);
}

fan::graphics::image_t loco_t::create_noise_image(const fan::vec2& size, const std::vector<uint8_t>& data) {
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

#if defined(loco_cuda)
void loco_t::cuda_textures_t::close(loco_t* loco, fan::graphics::shapes::shape_t& cid) {
  loco_t::universal_image_renderer_t::ri_t& ri = *(loco_t::universal_image_renderer_t::ri_t*)cid.GetData(fan::graphics::g_shapes->shaper);
  uint8_t image_amount = fan::graphics::get_channel_amount(ri.format);
  for (uint32_t i = 0; i < image_amount; ++i) {
    wresources[i].close();
    if (ri.images_rest[i] != loco->default_texture) {
      gloco()->image_unload(ri.images_rest[i]);
    }
    ri.images_rest[i] = loco->default_texture;
  }
  inited = false;
}

void loco_t::cuda_textures_t::resize(loco_t* loco, fan::graphics::shapes::shape_t& id, uint8_t format, fan::vec2ui size) {
  auto vi_image = id.get_image();
  if (vi_image.iic() || vi_image == loco->default_texture) {
    id.reload(format, size);
  }
  auto& ri = *(universal_image_renderer_t::ri_t*)id.GetData(loco->shaper);
  if (inited == false) {
    id.reload(format, size);
    vi_image = id.get_image();
    uint8_t image_amount = fan::graphics::get_channel_amount(format);
    for (uint32_t i = 0; i < image_amount; ++i) {
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
    for (uint32_t i = 0; i < fan::graphics::get_channel_amount(ri.format); ++i) {
      wresources[i].close();
    }
    id.reload(format, size);
    vi_image = id.get_image();
    ri = *(universal_image_renderer_t::ri_t*)id.GetData(loco->shaper);
    uint8_t image_amount = fan::graphics::get_channel_amount(format);
    // Re-register with CUDA after successful reload
    for (uint32_t i = 0; i < image_amount; ++i) {
      if (i == 0) {
        wresources[i].open(gloco()->image_get_handle(vi_image));
      }
      else {
        wresources[i].open(gloco()->image_get_handle(ri.images_rest[i - 1]));
      }
    }
  }
}

loco_t::cudaArray_t& loco_t::cuda_textures_t::get_array(uint32_t index_t) {
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
  //fan::cuda::check_error(cudaGraphicsResourceSetMapFlags(resource, 0));
}
#endif

#if defined(FAN_2D)

void loco_t::camera_move_to(const fan::graphics::shapes::shape_t& shape, const fan::graphics::render_view_t& render_view) {
  camera_set_position(
    orthographic_render_view.camera,
    shape.get_position()
  );
}

void loco_t::camera_move_to(const fan::graphics::shapes::shape_t& shape) {
  camera_move_to(shape, orthographic_render_view);
}

void loco_t::camera_move_to_smooth(const fan::graphics::shapes::shape_t& shape, const fan::graphics::render_view_t& render_view) {
  fan::vec2 current = camera_get_position(render_view.camera);
  fan::vec2 target = shape.get_position();
  f32_t t = 0.1f;
  camera_set_position(
    orthographic_render_view.camera,
    current.lerp(target, t)
  );
}

void loco_t::camera_move_to_smooth(const fan::graphics::shapes::shape_t& shape) {
  camera_move_to_smooth(shape, orthographic_render_view);
}


bool loco_t::shader_update_fragment(uint16_t shape_type, const std::string& fragment) {
  auto shader_nr = shader_get_nr(shape_type);
  auto shader_data = shader_get_data(shape_type);
  shader_set_vertex(shader_nr, shader_data.svertex);
  shader_set_fragment(shader_nr, fragment);
  return shader_compile(shader_nr);
}
#endif

#if defined(FAN_GUI)
namespace fan::graphics::gui {
  void process_frame() {
    auto it = gloco()->gui_draw_cb.GetNodeFirst();
    while (it != gloco()->gui_draw_cb.dst) {
      gloco()->gui_draw_cb.StartSafeNext(it);
      gloco()->gui_draw_cb[it]();
      it = gloco()->gui_draw_cb.EndSafeNext();
    }
  }
  // fan_track_allocations() must be called in global scope before calling this function
  void render_allocations_plot() {
    using namespace fan::graphics;

    struct pause_state_t {
      bool paused;
    };

    static pause_state_t pause_state = {
      false
    };

    gui::checkbox("pause updates", &pause_state.paused);

    static std::vector<f32_t> allocation_sizes;
    static std::vector<fan::heap_profiler_t::memory_data_t> allocations;
    static f32_t max_y = 0;

    if (!pause_state.paused) {
      allocation_sizes.clear();
      allocations.clear();
      max_y = 0;

      auto &profiler = fan::heap_profiler_t::instance();

      std::vector<std::pair<void*, fan::heap_profiler_t::memory_data_t>> sorted_allocs;
      sorted_allocs.reserve(profiler.memory_map.size());

      for (auto const &kv : profiler.memory_map) {
        sorted_allocs.emplace_back(const_cast<void*>(kv.first), kv.second);
      }

      std::sort(sorted_allocs.begin(), sorted_allocs.end(),
        [](auto const &a, auto const &b) {
        return (uintptr_t)a.first < (uintptr_t)b.first;
      });

      for (auto const &[addr, data] : sorted_allocs) {
        f32_t v = (f32_t)data.n / (1024 * 1024);
        allocation_sizes.push_back(v);
        max_y = max_y < v ? v : max_y;
        allocations.push_back(data);
      }
    }

    auto &profiler = fan::heap_profiler_t::instance();
    gui::text("Active allocations:", profiler.memory_map.size());
    gui::text("Allocation size:", profiler.current_allocation_size / 1e6, " (MB)");

    int total_mem_MB, used_MB;
    gloco()->get_vram_usage(&total_mem_MB, &used_MB);

    if (used_MB != -1) {
      gui::text("VRAM used memory", used_MB, " (MB)");
    }
    if (total_mem_MB != -1) {
      gui::text("VRAM total memory", total_mem_MB, " (MB)");
    }

    fan::vec2 cursor_pos = gui::get_cursor_pos();
    fan::vec2 window_size = gui::get_window_size();
    fan::vec2 available_size = window_size - cursor_pos;

  #if defined(fan_std23)
    static std::stacktrace stack;
  #endif

    if (allocation_sizes.size() && gui::plot::begin_plot("Memory Allocations", available_size, gui::plot::flags_no_frame | gui::plot::flags_no_legend)) {
      gui::plot::setup_axis(gui::plot::axis_y1, "Memory (MB)");
      gui::plot::setup_axis_limits(gui::plot::axis_y1, 0, max_y);
      gui::plot::setup_axis(gui::plot::axis_x1, "Allocations");
      gui::plot::setup_axis_limits(gui::plot::axis_x1, 0, (double)allocation_sizes.size());

      gui::plot::push_style_var(gui::plot::style_var_fill_alpha, 0.25f);
      gui::plot::plot_bars("Allocations", allocation_sizes.data(), allocation_sizes.size());
      gui::plot::pop_style_var();

      bool hovered = false;

      if (gui::plot::is_plot_hovered()) {
        auto mouse = gui::plot::get_plot_mouse_pos();
        mouse.x = (int)mouse.x;

        f32_t half_width = 0.25;
        f32_t tool_l = gui::plot::plot_to_pixels(mouse.x - half_width * 1.5, mouse.y).x;
        f32_t tool_r = gui::plot::plot_to_pixels(mouse.x + half_width * 1.5, mouse.y).x;
        f32_t tool_t = gui::plot::get_plot_pos().y;
        f32_t tool_b = tool_t + gui::plot::get_plot_size().y;

        gui::plot::push_plot_clip_rect();
        auto draw_list = gui::get_window_draw_list();
        draw_list->AddRectFilled(fan::vec2(tool_l, tool_t), fan::vec2(tool_r, tool_b),
          fan::color(128, 128, 128, 64).get_gui_color());
        gui::plot::pop_plot_clip_rect();

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
          if (end == std::string::npos) {
            break;
          }
          end += 1;
          auto begin = stack_str.rfind('\\', end);
          if (begin == std::string::npos) {
            break;
          }
          begin += 1;
          final_str += stack_str.substr(begin, end - begin);
          final_str += "\n";
          pos = end + 1;
        }

        text_unformatted(final_str.c_str());
        end_tooltip();
      }

      if (begin_popup("view stack", gui::window_flags_always_horizontal_scrollbar)) {
        std::ostringstream oss;
        oss << stack;
        text_unformatted(oss.str().c_str());
        end_popup();
      }
    #endif

      gui::plot::end_plot();
    }
  }
} // namespace fan::graphics::gui
#endif

void fan::graphics::shader_set_camera(fan::graphics::shader_t nr, fan::graphics::camera_t camera_nr) {
  if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::opengl) {
    get_gl_context().shader_set_camera(nr, camera_nr);
  }
#if defined(FAN_VULKAN)
  else if (fan::graphics::get_window().renderer == fan::window_t::renderer_t::vulkan) {
    fan::throw_error("todo");
  }
#endif
}

namespace fan {
  fan::event::task_t color_transition_t::animate(std::function<void(fan::color)> callback) {
    f32_t elapsed = 0;

    do {
      fan::time::timer frame_timer;
      frame_timer.start();

      while (elapsed < duration) {
        f32_t t = fmod((elapsed / duration) + phase_offset, 1.0f);

        switch (easing) {
        case ease_e::linear:
          break;
        case ease_e::sine:
          t = (std::sin((t - 0.5f) * fan::math::pi) + 1.f) * 0.5f;
          break;
        case ease_e::pulse:
          t = std::sin(t * fan::math::pi);
          break;
        case ease_e::ease_in:
          t = t * t;
          break;
        case ease_e::ease_out:
          t = 1.f - (1.f - t) * (1.f - t);
          break;
        }

        callback(from.lerp(to, t));
        co_await fan::graphics::co_next_frame();
        elapsed += gloco()->delta_time;
      }

      elapsed = 0;
      co_await fan::graphics::co_next_frame();

    } while (loop);

    callback(to);
    on_complete();
  }

  void auto_color_transition_t::start(
    const fan::color& from,
    const fan::color& to,
    f32_t duration,
    std::function<void(fan::color)> cb
  ) {
    if (active) {
      return;
    }

    fan::color_transition_t t;
    t.from = from;
    t.to = to;
    t.duration = duration;
    t.phase_offset = fan::random::value(0.0f, 1.0f);
    t.loop = true;
    t.easing = fan::color_transition_t::ease_e::pulse;

    transition = t;
    callback = cb;
    task = transition.animate(callback);
    active = true;
  }
  void auto_color_transition_t::start_once(
    const fan::color& from,
    const fan::color& to,
    f32_t duration,
    std::function<void(fan::color)> cb,
    std::function<void()> on_complete
  ) {
    if (active) return;
    transition.from = from;
    transition.to = to;
    transition.duration = duration;
    transition.phase_offset = 0.0f;
    transition.loop = false;
    transition.easing = fan::color_transition_t::ease_e::linear;
    callback = cb;
    transition.on_complete = on_complete;
    task = transition.animate(callback);
    active = true;
  }

  void auto_color_transition_t::stop(const fan::color& reset_to) {
    if (!active) return;
    task = {};
    callback(reset_to);
    active = false;
  }

  fan::color_transition_t fan::pulse_red(f32_t duration) {
    fan::color_transition_t t;
    t.from = fan::colors::white;
    t.to = fan::color(1, 0.2f, 0.2f);
    t.duration = duration;
    t.phase_offset = 0.0f;
    t.loop = true;
    t.easing = fan::color_transition_t::ease_e::pulse;
    return t;
  }
  fan::color_transition_t fan::fade_out(f32_t duration) {
    fan::color_transition_t t;
    t.from = fan::colors::white;
    t.to = fan::colors::transparent;
    t.duration = duration;
    t.phase_offset = 0.0f;
    t.loop = false;
    t.easing = fan::color_transition_t::ease_e::ease_out;
    return t;
  }
}