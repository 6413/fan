module;

#if defined(FAN_GUI)
  #include <fan/utility.h>

  #include <filesystem>
  #include <string>
  #include <functional>
  #include <coroutine>
  #include <cmath>
  #include <array>
  #include <vector>
  #include <unordered_map>

  #include <fan/imgui/imgui_internal.h>
#endif

module fan.graphics.gui;

#if defined(FAN_GUI)

import fan.utility;

#if defined(FAN_JSON)
import fan.types.json;
#endif

import fan.types.fstring;

import fan.types.vector;

import fan.print.error;

import fan.graphics.image_load;
import fan.graphics.gui.base;
import fan.graphics.gui.text_logger;
import fan.graphics; // fan::graphics::shader_get_data

import fan.file_dialog;

import fan.math;

#if defined(FAN_AUDIO)
  import fan.audio;
#endif

import fan.io.file;

import fan.crypto;

namespace fan::graphics::gui {
  const char* item_getter1(const std::vector<std::string>& items, int index) {
    if (index >= 0 && index < (int)items.size()) {
      return items[index].c_str();
    }
    return "N/A";
  }

  void set_viewport(fan::graphics::viewport_t viewport) {
    auto* current = get_current_window();
    viewport_rect_t main_viewport = get_viewport_rect();
    fan::vec2 wnd_pos = get_window_pos();
    fan::vec2 wnd_size = get_window_size();

    fan::vec2 windowPosRelativeToMainViewport;
    windowPosRelativeToMainViewport.x = wnd_pos.x - main_viewport.position.x;
    windowPosRelativeToMainViewport.y = wnd_pos.y - main_viewport.position.y;

    fan::vec2 viewport_size = fan::vec2(wnd_size.x, wnd_size.y);
    fan::vec2 viewport_pos = windowPosRelativeToMainViewport;

    fan::vec2 window_size = fan::graphics::get_window().get_size();
    fan::graphics::viewport_set(
      viewport,
      viewport_pos,
      viewport_size
    );
  }
  void set_viewport_full(fan::graphics::viewport_t viewport) {
    auto* current = get_current_window();
    viewport_rect_t main_viewport = get_viewport_rect();
    fan::vec2 wnd_pos = main_viewport.position;
    fan::vec2 wnd_size = main_viewport.size;
    if (current->ParentWindow) {
      wnd_pos = get_window_pos();
      wnd_size = get_window_size();
    }

    fan::vec2 windowPosRelativeToMainViewport;
    windowPosRelativeToMainViewport.x = wnd_pos.x - main_viewport.position.x;
    windowPosRelativeToMainViewport.y = wnd_pos.y - main_viewport.position.y;

    fan::vec2 viewport_size = fan::vec2(wnd_size.x, wnd_size.y);
    fan::vec2 viewport_pos = windowPosRelativeToMainViewport;

    fan::vec2 window_size = fan::graphics::get_window().get_size();
    fan::graphics::viewport_set(
      viewport,
      viewport_pos,
      viewport_size
    );
  }

  void set_viewport(const fan::graphics::render_view_t& render_view) {
    set_viewport(render_view.viewport);
    auto& cam = fan::graphics::camera_get(render_view.camera);

    if (!cam.original_coordinates[0] && !cam.original_coordinates[1] && !cam.original_coordinates[2] && !cam.original_coordinates[3]) {
      cam.original_coordinates = cam.coordinates.v;
    }

    fan::vec2 win = get_window_size();
    f32_t orig_w = cam.original_coordinates.y - cam.original_coordinates.x;
    f32_t orig_h = cam.original_coordinates.w - cam.original_coordinates.z;
    f32_t sx = win.x / orig_w;
    f32_t sy = win.y / orig_h;

    fan::graphics::camera_set_ortho(
      render_view.camera,
      fan::vec2(cam.original_coordinates.x * sx, cam.original_coordinates.y * sx),
      fan::vec2(cam.original_coordinates.z * sy, cam.original_coordinates.w * sy)
    );
  }

  bool toggle_image_button(fan::graphics::image_t* images, uint32_t count, const fan::vec2& size, int* selectedIndex) {
    f32_t y_pos = get_cursor_pos_y() + get_style().WindowPadding.y - get_style().FramePadding.y / 2;

    bool clicked = false;
    bool pushed = false;

    for (std::size_t i = 0; i < count; ++i) {
      fan::color tintColor = fan::color(0.2, 0.2, 0.2, 1.0);
      if (*selectedIndex == i) {
        tintColor = fan::color(0.2, 0.2, 0.2, 1.0f);
        push_style_color(col_button, tintColor);
        pushed = true;
      }
      /*if (ImGui::IsItemHovered()) {
      tintColor = fan::color(1, 1, 1, 1.0f);
      }*/
      set_cursor_pos_y(y_pos);

      gui::push_id(i);
      gui::push_id(&clicked);
      if (fan::graphics::gui::image_button("##toggle", *(images + i), size)) {
        *selectedIndex = i;
        clicked = true;
      }
      gui::pop_id();
      gui::pop_id();
      if (pushed) {
        pop_style_color();
        pushed = false;
      }
      same_line();
    }

    return clicked;
  }

  bool toggle_image_button(fan::str_view_t char_id, fan::graphics::image_t image, const fan::vec2& size, bool* toggle) {
    bool clicked = false;

    fan::color tintColor = fan::color(1, 1, 1, 1);
    if (*toggle) {
      tintColor = fan::color(0.3f, 0.3f, 0.3f, 1.0f);
    }

    if (image_button(char_id, image, size, fan::vec2(0, 0), fan::vec2(1, 1), -1, fan::color(0, 0, 0, 0), tintColor)) {
      *toggle = !(*toggle);
      clicked = true;
    }

    return clicked;
  }

  // untested
  void image_rotated(fan::graphics::image_t image, const fan::vec2& size, int angle, const fan::vec2& uv0, const fan::vec2& uv1, const fan::color& tint_col, const fan::color& border_col) {
    if (!(angle % 90 == 0)) {
      fan::throw_error("invalid angle");
    }
    fan::vec2 _uv0, _uv1, _uv2, _uv3;

    switch (angle % 360) {
    case 0:
      gui::image(image, size, uv0, uv1, tint_col, border_col);
      return;
    case 180:
      gui::image(image, size, uv1, uv0, tint_col, border_col);
      return;
    case 90:
      _uv3 = uv0;
      _uv1 = uv1;
      _uv0 = fan::vec2(uv1.x, uv0.y);
      _uv2 = fan::vec2(uv0.x, uv1.y);
      break;
    case 270:
      _uv1 = uv0;
      _uv3 = uv1;
      _uv0 = fan::vec2(uv0.x, uv1.y);
      _uv2 = fan::vec2(uv1.x, uv0.y);
      break;
    }

    window_handle_t* window = get_current_window();
    if (window->SkipItems)
      return;

    fan::vec2 _size(size.y, size.x); // swapped for rotation
    fan::vec2 cursor_pos = *(fan::vec2*)&window->DC.CursorPos;
    fan::vec2 bb_max = cursor_pos + _size;
    if (border_col.a > 0.0f) {
      bb_max += fan::vec2(2, 2);
    }

    rect_t bb(*(fan::vec2*)&cursor_pos, *(fan::vec2*)&bb_max);
    item_size(bb);
    if (!item_add(bb, 0))
      return;

    if (border_col.a > 0.0f) {
      window->DrawList->AddRect(*(fan::vec2*)&bb.Min, *(fan::vec2*)&bb.Max, border_col.get_gui_color(), 0.0f);
      fan::vec2 x0 = cursor_pos + fan::vec2(1, 1);
      fan::vec2 x2 = bb_max - fan::vec2(1, 1);
      fan::vec2 x1 = fan::vec2(x2.x, x0.y);
      fan::vec2 x3 = fan::vec2(x0.x, x2.y);

      window->DrawList->AddImageQuad(
        (texture_id_t)fan::graphics::image_get_handle(image),
        *(fan::vec2*)&x0, *(fan::vec2*)&x1, *(fan::vec2*)&x2, *(fan::vec2*)&x3,
        *(fan::vec2*)&_uv0, *(fan::vec2*)&_uv1, *(fan::vec2*)&_uv2, *(fan::vec2*)&_uv3,
        tint_col.get_gui_color()
      );
    }
    else {
      fan::vec2 x0 = cursor_pos;
      fan::vec2 x1 = fan::vec2(bb_max.x, cursor_pos.y);
      fan::vec2 x2 = bb_max;
      fan::vec2 x3 = fan::vec2(cursor_pos.x, bb_max.y);

      window->DrawList->AddImageQuad(
        (texture_id_t)fan::graphics::image_get_handle(image),
        *(fan::vec2*)&x0, *(fan::vec2*)&x1, *(fan::vec2*)&x2, *(fan::vec2*)&x3,
        *(fan::vec2*)&_uv0, *(fan::vec2*)&_uv1, *(fan::vec2*)&_uv2, *(fan::vec2*)&_uv3,
        tint_col.get_gui_color()
      );
    }
  }

  imgui_element_nr_t::imgui_element_nr_t(const imgui_element_nr_t& nr) : imgui_element_nr_t() {
    if (nr.is_invalid()) {
      return;
    }
    init();
  }

  imgui_element_nr_t::imgui_element_nr_t(imgui_element_nr_t&& nr) {
    NRI = nr.NRI;
    nr.invalidate_soft();
  }

  imgui_element_nr_t::~imgui_element_nr_t() {
    invalidate();
  }

  imgui_element_nr_t& imgui_element_nr_t::operator=(const imgui_element_nr_t& id) {
    if (!is_invalid()) {
      invalidate();
    }
    if (id.is_invalid()) {
      return *this;
    }

    if (this != &id) {
      init();
    }
    return *this;
  }

  imgui_element_nr_t& imgui_element_nr_t::operator=(imgui_element_nr_t&& id) {
    if (!is_invalid()) {
      invalidate();
    }
    if (id.is_invalid()) {
      return *this;
    }

    if (this != &id) {
      if (!is_invalid()) {
        invalidate();
      }
      NRI = id.NRI;

      id.invalidate_soft();
    }
    return *this;
  }

  void imgui_element_nr_t::init() {
    *(base_t*)this = fan::graphics::get_gui_draw_cbs().NewNodeLast();
  }

  bool imgui_element_nr_t::is_invalid() const {
    return fan::graphics::gui_draw_cb_inric(*this);
  }

  void imgui_element_nr_t::invalidate_soft() {
    *(base_t*)this = fan::graphics::get_gui_draw_cbs().gnric();
  }

  void imgui_element_nr_t::invalidate() {
    if (is_invalid()) {
      return;
    }
    fan::graphics::get_gui_draw_cbs().unlrec(*this);
    *(base_t*)this = fan::graphics::get_gui_draw_cbs().gnric();
  }

  void imgui_element_nr_t::set(const std::function<void()>& lambda) {
    fan::graphics::get_gui_draw_cbs()[*this] = lambda;
  }

  imgui_element_t::imgui_element_t(const std::function<void()>& lambda) {
    imgui_element_nr_t::init();
    imgui_element_nr_t::set(lambda);
  }

#if defined(FAN_2D)
  void shape_properties(fan::graphics::shape_t& shape) {
    if (!shape.get_visual_id()) {
      return;
    }
    switch (shape.get_shape_type()) {
    case fan::graphics::shape_type_t::particles:
    {
      auto& ri = *(fan::graphics::shapes::particles_t::ri_t*)shape.GetData(fan::graphics::g_shapes->shaper);
      static std::unordered_map<void*, bool> prev_loop_states;
      bool& prev_loop_state = prev_loop_states[&ri];

      if (checkbox("loop", &ri.loop)) {
        if (prev_loop_state && !ri.loop) {
          shape.stop_particles();
        }
        if (!prev_loop_state && ri.loop) {
          shape.start_particles();
        }
        prev_loop_state = ri.loop;
      }

      const char* items[] = {"circle", "rectangle"};
      int current_shape = ri.shape;
      if (combo("shape", &current_shape, items, std::size(items))) {
        ri.shape = current_shape;
      }
      if (color_edit4("particle begin color", &ri.begin_color)) {
      }
      if (color_edit4("particle end color", &ri.end_color)) {
      }
      if (drag("position", &ri.position, 1)) {
      }
      if (drag("start_size", &ri.start_size, 1)) {
      }
      if (drag("end_size", &ri.end_size, 1)) {
      }
      if (drag("alive_time", &ri.alive_time, 0.01)) {
      }
      if (ri.shape == fan::graphics::shapes::particles_t::shapes_e::rectangle) {
        if (drag("spawn_spacing", &ri.spawn_spacing, 1)) {
        }
      }
      if (drag("expansion_power", &ri.expansion_power, 0.01)) {
      }
      if (drag("start_spread", &ri.start_spread, 0.1)) {
      }
      if (drag("end_spread", &ri.end_spread, 0.1)) {
      }
      if (drag("start_velocity", &ri.start_velocity, 0.1)) {
      }
      if (drag("end_velocity", &ri.end_velocity, 0.1)) {
      }
      if (drag("start_angle_velocity", &ri.start_angle_velocity, 0.1)) {
      }
      if (drag("end_angle_velocity", &ri.end_angle_velocity, 0.1)) {
      }
      if (drag("jitter_start", &ri.jitter_start, 0.1)) {
      }
      if (drag("jitter_end", &ri.jitter_end, 0.1)) {
      }
      if (drag("jitter_speed", &ri.jitter_speed, 0.1)) {
      }
      if (drag("size_random_range", &ri.size_random_range, 0.1)) {
      }
      if (drag("color_random_range", &ri.color_random_range, 0.1)) {
      }
      if (drag("angle_random_range", &ri.angle_random_range, 0.1)) {
      }
      if (drag("count", &ri.count, 1)) {
      }
      if (slider("begin_angle", &ri.begin_angle, -fan::math::pi * 2, fan::math::pi * 2)) {
      }
      if (slider("end_angle", &ri.end_angle, -fan::math::pi * 2, fan::math::pi * 2)) {
      }
      if (slider("angle", &ri.angle, -fan::math::pi * 2, fan::math::pi * 2)) {
      }
      g_shapes->visit_shape_draw_data(shape.gint(), [&]<typename T>(T & properties) {
        if constexpr (std::is_same_v<fan::graphics::shapes::particles_t::properties_t, T>) {
          properties.loop = ri.loop;
          properties.shape = ri.shape;
          properties.begin_color = ri.begin_color;
          properties.end_color = ri.end_color;
          properties.position = ri.position;
          properties.start_size = ri.start_size;
          properties.end_size = ri.end_size;
          properties.alive_time = ri.alive_time;
          properties.spawn_spacing = ri.spawn_spacing;
          properties.expansion_power = ri.expansion_power;
          properties.start_spread = ri.start_spread;
          properties.end_spread = ri.end_spread;
          properties.start_velocity = ri.start_velocity;
          properties.end_velocity = ri.end_velocity;
          properties.start_angle_velocity = ri.start_angle_velocity;
          properties.end_angle_velocity = ri.end_angle_velocity;
          properties.jitter_start = ri.jitter_start;
          properties.jitter_end = ri.jitter_end;
          properties.jitter_speed = ri.jitter_speed;
          properties.count = ri.count;
          properties.begin_angle = ri.begin_angle;
          properties.end_angle = ri.end_angle;
          properties.angle = ri.angle;
        }
      });

      break;
    }
    }
  }

#endif

  static std::string path_join(std::string_view a, std::string_view b) {
    if (a.empty()) return std::string(b);
    if (b.empty()) return std::string(a);
    std::string r(a);
    if (r.back() != '/' && r.back() != '\\') r += '/';
    r += b;
    return r;
  }

  static std::string path_filename(std::string_view p) {
    auto pos = p.find_last_of("/\\");
    return pos == std::string_view::npos ? std::string(p) : std::string(p.substr(pos + 1));
  }

  static std::string path_parent(std::string_view p) {
    auto end = p.size();
    while (end > 1 && (p[end - 1] == '/' || p[end - 1] == '\\')) --end;
    auto pos = p.find_last_of("/\\", end - 1);
    if (pos == std::string_view::npos) return ".";
    return pos == 0 ? "/" : std::string(p.substr(0, pos));
  }

  static std::string path_relative(std::string_view path, std::string_view base) {
    if (path.substr(0, base.size()) == base) {
      auto rel = path.substr(base.size());
      if (!rel.empty() && (rel[0] == '/' || rel[0] == '\\')) rel = rel.substr(1);
      return std::string(rel);
    }
    return std::string(path);
  }

  static bool is_absolute(std::string_view p) {
    if (p.size() >= 2 && p[1] == ':') return true;
    if (!p.empty() && (p[0] == '/' || p[0] == '\\')) return true;
    return false;
  }

  content_browser_t::content_browser_t() {
    search_buffer.resize(32);
    current_directory = asset_path;
    update_directory_cache();
  }

  content_browser_t::content_browser_t(bool no_init) {}

  content_browser_t::content_browser_t(const std::string& path) {
    init(path);
  }

  void content_browser_t::init(const std::string& path) {
    search_buffer.resize(32);
    current_directory = path_join(asset_path, path);
    update_directory_cache();
  }

  void content_browser_t::clear_selection() {
    selection_state.selected_indices.clear();
    for (auto& file : directory_cache) file.is_selected = false;
    for (auto& file : search_state.found_files) file.is_selected = false;
  }

  bool content_browser_t::is_point_in_rect(const fan::vec2& point, const fan::vec2& rect_min, const fan::vec2& rect_max) {
    return point.x >= rect_min.x && point.x <= rect_max.x &&
      point.y >= rect_min.y && point.y <= rect_max.y;
  }

  void content_browser_t::handle_rectangular_selection() {
    io_t& io = get_io();
    selection_state.ctrl_held = io.KeyCtrl;

    if (fan::window::is_mouse_clicked()) {
      bool can_start_selection = !is_any_item_active() && is_window_hovered() && !is_any_item_hovered();
      if (can_start_selection) {
        selection_state.is_selecting = true;
        selection_state.selection_start = get_mouse_pos();
        selection_state.selection_end = selection_state.selection_start;
        clear_selection();
      }
    }

    if (selection_state.is_selecting && fan::window::is_mouse_down()) {
      selection_state.selection_end = get_mouse_pos();
      bool showing_search_results = !search_state.found_files.empty() && !search_buffer.empty();
      if (showing_search_results) update_search_sorted_cache();
      else update_sorted_cache();
    }

    if (selection_state.is_selecting && fan::window::is_mouse_released()) {
      selection_state.is_selecting = false;
    }

    if (selection_state.is_selecting) {
      draw_list_t* draw_list = get_window_draw_list();
      fan::vec2 rect_min = fan::vec2(
        std::min(selection_state.selection_start.x, selection_state.selection_end.x),
        std::min(selection_state.selection_start.y, selection_state.selection_end.y)
      );
      fan::vec2 rect_max = fan::vec2(
        std::max(selection_state.selection_start.x, selection_state.selection_end.x),
        std::max(selection_state.selection_start.y, selection_state.selection_end.y)
      );
      draw_list->AddRect(rect_min, rect_max, fan::color(100, 150, 255, 200).get_gui_color(), 0.0f, 0, 2.0f);
      draw_list->AddRectFilled(rect_min, rect_max, fan::color(100, 150, 255, 50).get_gui_color());
    }
  }

  void content_browser_t::update_directory_cache() {
    search_iterator.stop();
    search_state.is_searching = false;
    search_state.found_files.clear();
    search_state.search_cache_dirty = true;
    search_state.cache_dirty = true;
    while (!search_state.pending_directories.empty()) search_state.pending_directories.pop();

    std::fill(search_buffer.begin(), search_buffer.end(), '\0');
    clear_selection();
    invalidate_cache();

    for (auto& img : directory_cache) {
      if (fan::graphics::is_image_valid(img.preview_image)) {
        fan::graphics::image_unload(img.preview_image);
        img.preview_image.sic();
      }
    }
    directory_cache.clear();

    if (!directory_iterator.callback) {
      directory_iterator.sort_alphabetically = true;
      directory_iterator.callback = [this](const std::string& path_str, bool is_dir) -> fan::event::task_t {
        file_info_t file_info;
        file_info.filename = path_filename(path_str);
        file_info.item_path = path_relative(path_str, asset_path);
        file_info.is_directory = is_dir;
        file_info.is_selected = false;

        if (!is_dir && fan::image::valid(path_str)) {
          file_info.preview_image = fan::graphics::image_load(path_str);
        }

        directory_cache.push_back(file_info);
        invalidate_cache();
        co_return;
      };
    }

    fan::io::async_directory_iterate(&directory_iterator, current_directory);
  }

  void content_browser_t::invalidate_cache() {
    search_state.cache_dirty = true;
    search_state.sorted_cache.clear();
  }

  int content_browser_t::get_pressed_key() {
    return gui::get_pressed_key();
  }

  void content_browser_t::handle_keyboard_navigation(std::string_view filename, int pressed_key) {
    if (pressed_key != -1 && !filename.empty()) {
      if (std::tolower(filename[0]) == std::tolower(pressed_key)) {
        set_scroll_here_y();
      }
    }
  }

  void content_browser_t::handle_right_click(std::string_view filename) {
    if (is_item_hovered() && fan::window::is_mouse_clicked(fan::mouse_right)) {
      item_right_clicked = true;
      item_right_clicked_name = filename;
      auto start = item_right_clicked_name.find_first_not_of(" \t\r\n");
      if (start != std::string::npos) item_right_clicked_name = item_right_clicked_name.substr(start);
      auto end = item_right_clicked_name.find_last_not_of(" \t\r\n");
      if (end != std::string::npos) item_right_clicked_name = item_right_clicked_name.substr(0, end + 1);
    }
  }

  void content_browser_t::process_next_directory() {
    if (!search_state.pending_directories.empty()) {
      std::string next_dir = search_state.pending_directories.front();
      search_state.pending_directories.pop();
      fan::io::async_directory_iterate(&search_iterator, next_dir);
    }
    else {
      search_state.is_searching = false;
    }
  }

  void content_browser_t::start_search(const std::string& query, bool recursive) {
    search_iterator.stop();
    search_state.query = query;
    search_state.is_recursive = recursive;

    for (auto& img : search_state.found_files) {
      if (fan::graphics::is_image_valid(img.preview_image)) {
        fan::graphics::image_unload(img.preview_image);
        img.preview_image.sic();
      }
    }
    search_state.found_files.clear();
    search_state.search_cache_dirty = true;
    while (!search_state.pending_directories.empty()) search_state.pending_directories.pop();

    if (query.empty()) {
      search_state.is_searching = false;
      return;
    }

    search_state.is_searching = true;
    search_iterator.sort_alphabetically = true;
    search_iterator.callback = [this](const std::string& path_str, bool is_dir) -> fan::event::task_t {
      try {
        std::string filename = path_filename(path_str);
        std::string query_lower = search_state.query;
        std::string filename_lower = filename;
        static auto tl = [](unsigned char c) { return std::tolower(c); };
        std::transform(query_lower.begin(), query_lower.end(), query_lower.begin(), tl);
        std::transform(filename_lower.begin(), filename_lower.end(), filename_lower.begin(), tl);

        if (filename_lower.find(query_lower) != std::string::npos) {
          file_info_t file_info;
          file_info.filename = filename;
          file_info.item_path = path_relative(path_str, asset_path);
          file_info.is_directory = is_dir;
          file_info.is_selected = false;

          if (fan::image::valid(path_str)) {
            file_info.preview_image = fan::graphics::image_load(path_str);
          }

          search_state.found_files.push_back(file_info);
          search_state.search_cache_dirty = true;
        }

        if (is_dir && search_state.is_recursive) {
          search_state.pending_directories.push(path_str);
        }
      }
      catch (...) {}
      co_return;
    };

    fan::io::async_directory_iterate(&search_iterator, current_directory);
  }

  void content_browser_t::update_sorted_cache() {
    if (!search_state.cache_dirty) return;
    search_state.sorted_cache.clear();
    for (std::size_t i = 0; i < directory_cache.size(); ++i) {
      auto& file_info = directory_cache[i];
      if (search_buffer.empty() || file_info.filename.find(search_buffer.data()) != std::string::npos) {
        search_state.sorted_cache.push_back({file_info, i});
      }
    }
    std::sort(search_state.sorted_cache.begin(), search_state.sorted_cache.end(),
      [](const auto& a, const auto& b) { return a.first.is_directory > b.first.is_directory; });
    search_state.cache_dirty = false;
  }

  void content_browser_t::update_search_sorted_cache() {
    if (!search_state.search_cache_dirty) return;
    search_state.sorted_search_cache.clear();
    for (std::size_t i = 0; i < search_state.found_files.size(); ++i) {
      search_state.sorted_search_cache.push_back({search_state.found_files[i], i});
    }
    std::sort(search_state.sorted_search_cache.begin(), search_state.sorted_search_cache.end(),
      [](const auto& a, const auto& b) { return a.first.is_directory > b.first.is_directory; });
    search_state.search_cache_dirty = false;
  }

  void content_browser_t::render() {
    item_right_clicked = false;
    item_right_clicked_name.clear();

    if (search_state.is_searching && !search_iterator.operation_in_progress) {
      if (search_iterator.is_finished()) {
        if (!search_state.pending_directories.empty()) process_next_directory();
        else search_state.is_searching = false;
      }
    }

    style_t& style = get_style();
    push_style_var(style_var_frame_padding, fan::vec2(10.0f, 16.0f));

    class_t window_class;
    set_next_window_class(&window_class);

    if (begin("Content Browser", nullptr, window_flags_menu_bar | window_flags_no_title_bar)) {
      if (begin_menu_bar()) {
        push_style_color(col_button, fan::color(0.f, 0.f, 0.f, 0.f));
        push_style_color(col_button_active, fan::color(0.f, 0.f, 0.f, 0.f));
        push_style_color(col_button_hovered, fan::color(0.3f, 0.3f, 0.3f, 0.3f));

        if (image_button("##icon_arrow_left", icon_arrow_left, fan::vec2(32))) {
          current_directory = path_parent(current_directory);
          update_directory_cache();
        }
        same_line();
        
        if (button("Open..")) {
          fan::graphics::open_folder([this](std::string_view path) {
             current_directory = std::string(path);
             update_directory_cache();
          });
        }
        same_line();

        push_style_var(style_var_frame_padding, fan::vec2(10.0f, 7.0f));
        set_cursor_pos_y(get_cursor_pos_y() + get_style().WindowPadding.y);
        
        fan::vec2 button_sizes = 32;
        f32_t right_aligned_elements_width = 300.0f + (button_sizes.x * 2 + style.ItemSpacing.x * 2); 
        set_next_item_width(get_content_region_avail().x - right_aligned_elements_width);
        
        if (input_text("##current_directory_input", &current_directory, input_text_flags_enter_returns_true)) {
           update_directory_cache();
        }
        pop_style_var();

        pop_style_color(3);

        push_style_color(col_button, fan::color(0.f, 0.f, 0.f, 0.f));
        push_style_color(col_button_active, fan::color(0.f, 0.f, 0.f, 0.f));
        push_style_color(col_button_hovered, fan::color(0.3f, 0.3f, 0.3f, 0.3f));

        auto image_list = std::to_array({icon_files_list, icon_files_big_thumbnail});
        fan::vec2 bc = get_position_bottom_corner();
        bc.x -= get_window_pos().x;
        
        set_cursor_pos_x(get_window_size().x - right_aligned_elements_width);

        set_next_item_width(300.0f);

        push_style_var(style_var_frame_rounding, 20.0f);
        push_style_var(style_var_frame_padding, fan::vec2(10.0f, 7.0f));
        f32_t y_pos = get_cursor_pos_y() + get_style().WindowPadding.y;
        set_cursor_pos_y(y_pos);

        if (input_text("##content_browser_search", &search_buffer)) {
          start_search(search_buffer.data(), true);
        }

        if (search_state.is_searching) {
          same_line();
          text("Searching... (%zu found)", search_state.found_files.size());
        }

        pop_style_var(2);
        
        same_line();
        toggle_image_button(image_list.data(), image_list.size(), button_sizes, (int*)&current_view_mode);
        
        pop_style_color(3);
        end_menu_bar();
      }

      handle_rectangular_selection();

      switch (current_view_mode) {
      case view_mode_large_thumbnails: render_large_thumbnails_view(); break;
      case view_mode_list: render_list_view(); break;
      default: break;
      }
    }

    pop_style_var();
    gui::end();

    if (!pending_directory_change.empty()) {
      current_directory = pending_directory_change;
      update_directory_cache();
      pending_directory_change.clear();
    }
  }
  void content_browser_t::handle_item_interaction(const file_info_t& file_info, size_t original_index) {
    if (!file_info.is_directory) {
      if (begin_drag_drop_source()) {
        bool showing_search_results = !search_state.found_files.empty() && search_buffer[0] != '\0';

        if (showing_search_results) {
          if (!search_state.found_files[original_index].is_selected) {
            for (auto& f : search_state.found_files) f.is_selected = false;
            search_state.found_files[original_index].is_selected = true;
          }
        }
        else {
          if (!directory_cache[original_index].is_selected) {
            for (auto& f : directory_cache) f.is_selected = false;
            directory_cache[original_index].is_selected = true;
          }
        }

        std::string combined_paths;
        auto& src = showing_search_results ? search_state.found_files : directory_cache;
        for (const auto& f : src) {
          if (f.is_selected && !f.is_directory) {
            if (!combined_paths.empty()) combined_paths += ';';
            combined_paths += f.item_path;
          }
        }
        if (combined_paths.empty()) combined_paths = file_info.item_path;

        set_drag_drop_payload("CONTENT_BROWSER_ITEMS", combined_paths.data(), combined_paths.size() + 1);

        auto count = std::count(combined_paths.begin(), combined_paths.end(), ';') + 1;
        if (count > 1) text(count, " files selected");
        else text(file_info.filename);

        end_drag_drop_source();
      }
    }

    if (is_item_hovered() && is_mouse_double_clicked(0)) {
      if (file_info.is_directory) {
        pending_directory_change = path_join(asset_path, file_info.item_path);
      }
    }
  }

  void content_browser_t::receive_drag_drop_target(std::function<void(const std::string&)> receive_func) {
    dummy(get_content_region_avail());

    if (begin_drag_drop_target()) {
      if (const payload_t* payload = accept_drag_drop_payload("CONTENT_BROWSER_ITEMS")) {
        std::string combined_paths(static_cast<const char*>(payload->Data));
        std::string current_path;
        for (char c : combined_paths) {
          if (c == ';') {
            if (!current_path.empty()) {
              receive_func(is_absolute(current_path) ? current_path : path_join(asset_path, current_path));
              current_path.clear();
            }
          }
          else {
            current_path += c;
          }
        }
        if (!current_path.empty()) {
          receive_func(is_absolute(current_path) ? current_path : path_join(asset_path, current_path));
        }
      }
      end_drag_drop_target();
    }
  }

  void content_browser_t::render_large_thumbnails_view() {
    f32_t thumbnail_size = 128.0f;
    f32_t panel_width = get_content_region_avail().x;
    int column_count = std::max((int)(panel_width / (thumbnail_size + padding)), 1);

    columns(column_count, 0, false);
    int pressed_key = get_pressed_key();

    bool showing_search_results = !search_state.found_files.empty() && !search_buffer.empty();

    if (showing_search_results) {
      update_search_sorted_cache();
      auto& sorted_files = search_state.sorted_search_cache;

      for (const auto& [file_info, original_index] : sorted_files) {
        handle_keyboard_navigation(file_info.filename, pressed_key);

        gui::push_id(original_index);
        gui::push_id(file_info.filename.c_str());
        fan::vec2 item_pos = get_cursor_screen_pos();
        bool is_currently_selected = search_state.found_files[original_index].is_selected;

        if (is_currently_selected) {
          push_style_color(col_button, fan::color(0.4f, 0.6f, 1.0f, 0.3f));
          push_style_color(col_button_hovered, fan::color(0.4f, 0.6f, 1.0f, 0.4f));
          push_style_color(col_button_active, fan::color(0.4f, 0.6f, 1.0f, 0.5f));
        }
        else {
          push_style_color(col_button, fan::color(0, 0, 0, 0));
          push_style_color(col_button_hovered, fan::color(0.3f, 0.3f, 0.3f, 0.2f));
          push_style_color(col_button_active, fan::color(0.5f, 0.5f, 0.5f, 0.3f));
        }

        bool item_clicked = image_button(
          "##",
          fan::graphics::is_image_valid(file_info.preview_image) ? file_info.preview_image
          : file_info.is_directory ? icon_directory : file_info.filename.ends_with(".json") ? icon_object : icon_file,
          fan::vec2(thumbnail_size, thumbnail_size)
        );

        if (item_clicked) {
          auto& io = get_io();
          if (io.KeyCtrl) {
            search_state.found_files[original_index].is_selected = !search_state.found_files[original_index].is_selected;
          }
          else {
            for (auto& f : search_state.found_files) f.is_selected = false;
            search_state.found_files[original_index].is_selected = true;
          }
        }

        if (selection_state.is_selecting) {
          fan::vec2 rect_min = fan::vec2(
            std::min(selection_state.selection_start.x, selection_state.selection_end.x),
            std::min(selection_state.selection_start.y, selection_state.selection_end.y)
          );
          fan::vec2 rect_max = fan::vec2(
            std::max(selection_state.selection_start.x, selection_state.selection_end.x),
            std::max(selection_state.selection_start.y, selection_state.selection_end.y)
          );
          fan::vec2 item_min = item_pos;
          fan::vec2 item_max = fan::vec2(item_pos.x + thumbnail_size, item_pos.y + thumbnail_size);
          bool overlaps = !(rect_max.x < item_min.x || rect_min.x > item_max.x ||
            rect_max.y < item_min.y || rect_min.y > item_max.y);
          if (overlaps) search_state.found_files[original_index].is_selected = true;
          else if (!selection_state.ctrl_held) search_state.found_files[original_index].is_selected = false;
        }

        handle_right_click(file_info.filename);
        handle_item_interaction(file_info, original_index);
        pop_style_color(3);
        text_wrapped(file_info.filename);
        next_column();
        pop_id();
        pop_id();
      }
    }
    else {
      update_sorted_cache();
      auto& sorted_files = search_state.sorted_cache;

      for (const auto& [file_info, original_index] : sorted_files) {
        handle_keyboard_navigation(file_info.filename, pressed_key);
        push_id(std::string_view(file_info.filename));

        fan::vec2 item_pos = get_cursor_screen_pos();
        bool is_currently_selected = directory_cache[original_index].is_selected;

        if (is_currently_selected) {
          push_style_color(col_button, fan::color(0.4f, 0.6f, 1.0f, 0.3f));
          push_style_color(col_button_hovered, fan::color(0.4f, 0.6f, 1.0f, 0.4f));
          push_style_color(col_button_active, fan::color(0.4f, 0.6f, 1.0f, 0.5f));
        }
        else {
          push_style_color(col_button, fan::color(0, 0, 0, 0));
          push_style_color(col_button_hovered, fan::color(0.3f, 0.3f, 0.3f, 0.2f));
          push_style_color(col_button_active, fan::color(0.5f, 0.5f, 0.5f, 0.3f));
        }

        std::string id = "##" + file_info.filename;
        bool item_clicked = image_button(
          std::string_view(id),
          fan::graphics::is_image_valid(file_info.preview_image) ? file_info.preview_image
          : file_info.is_directory ? icon_directory : file_info.filename.ends_with(".json") ? icon_object : icon_file,
          fan::vec2(thumbnail_size, thumbnail_size)
        );

        if (item_clicked) {
          auto& io = get_io();
          if (io.KeyCtrl) {
            directory_cache[original_index].is_selected = !directory_cache[original_index].is_selected;
          }
          else {
            for (auto& f : directory_cache) f.is_selected = false;
            directory_cache[original_index].is_selected = true;
          }
        }

        if (selection_state.is_selecting) {
          fan::vec2 rect_min = fan::vec2(
            std::min(selection_state.selection_start.x, selection_state.selection_end.x),
            std::min(selection_state.selection_start.y, selection_state.selection_end.y)
          );
          fan::vec2 rect_max = fan::vec2(
            std::max(selection_state.selection_start.x, selection_state.selection_end.x),
            std::max(selection_state.selection_start.y, selection_state.selection_end.y)
          );
          fan::vec2 item_min = item_pos;
          fan::vec2 item_max = fan::vec2(item_pos.x + thumbnail_size, item_pos.y + thumbnail_size);
          bool overlaps = !(rect_max.x < item_min.x || rect_min.x > item_max.x ||
            rect_max.y < item_min.y || rect_min.y > item_max.y);
          if (overlaps) directory_cache[original_index].is_selected = true;
          else if (!selection_state.ctrl_held) directory_cache[original_index].is_selected = false;
        }

        handle_right_click(file_info.filename);
        handle_item_interaction(file_info, original_index);
        pop_style_color(3);
        text_wrapped(file_info.filename.c_str());
        next_column();
        pop_id();
      }
    }
    columns(1);
  }

  void content_browser_t::render_list_view() {
    if (begin_table("##FileTable", 1,
      table_flags_sizing_fixed_fit | table_flags_scroll_x | table_flags_scroll_y |
      table_flags_row_bg | table_flags_borders_outer | table_flags_borders_v |
      table_flags_resizable | table_flags_reorderable | table_flags_hideable |
      table_flags_sortable)) {

      table_setup_column("##Filename", table_column_flags_width_stretch);
      table_headers_row();

      int pressed_key = get_pressed_key();
      bool showing_search_results = !search_state.found_files.empty() && !search_buffer.empty();

      if (showing_search_results) {
        update_search_sorted_cache();
        auto& sorted_files = search_state.sorted_search_cache;

        for (const auto& [file_info, original_index] : sorted_files) {
          handle_keyboard_navigation(file_info.filename, pressed_key);

          table_next_row();
          table_set_column_index(0);

          fan::vec2 row_pos = get_cursor_screen_pos();
          f32_t row_height = get_text_line_height_with_spacing();
          fan::vec2 cursor_pos = fan::vec2(get_window_pos()) + fan::vec2(get_cursor_pos()) +
            fan::vec2(get_scroll_x(), -get_scroll_y());
          fan::vec2 image_size = fan::vec2(thumbnail_size / 4, thumbnail_size / 4);

          std::string space;
          while (calc_text_size(space.c_str()).x < image_size.x) space += " ";
          std::string unique_id = "search_" + std::to_string(original_index) + "_" + space + file_info.filename;

          bool item_clicked = selectable(unique_id.c_str(), search_state.found_files[original_index].is_selected, selectable_flags_span_all_columns);

          if (item_clicked) {
            auto& io = get_io();
            if (io.KeyCtrl) {
              search_state.found_files[original_index].is_selected = !search_state.found_files[original_index].is_selected;
            }
            else {
              for (auto& f : search_state.found_files) f.is_selected = false;
              search_state.found_files[original_index].is_selected = true;
            }
          }

          if (selection_state.is_selecting) {
            fan::vec2 rect_min = fan::vec2(
              std::min(selection_state.selection_start.x, selection_state.selection_end.x),
              std::min(selection_state.selection_start.y, selection_state.selection_end.y)
            );
            fan::vec2 rect_max = fan::vec2(
              std::max(selection_state.selection_start.x, selection_state.selection_end.x),
              std::max(selection_state.selection_start.y, selection_state.selection_end.y)
            );
            fan::vec2 row_min = row_pos;
            fan::vec2 row_max = fan::vec2(row_pos.x + get_content_region_avail().x, row_pos.y + row_height);
            bool overlaps = !(rect_max.x < row_min.x || rect_min.x > row_max.x ||
              rect_max.y < row_min.y || rect_min.y > row_max.y);
            if (overlaps) search_state.found_files[original_index].is_selected = true;
            else if (!selection_state.ctrl_held) search_state.found_files[original_index].is_selected = false;
          }

          handle_right_click(file_info.filename);

          texture_id_t texture_id = (texture_id_t)fan::graphics::image_get_handle(
            fan::graphics::is_image_valid(file_info.preview_image) ? file_info.preview_image
            : file_info.is_directory ? icon_directory : icon_file
          );
          get_window_draw_list()->AddImage(texture_id, cursor_pos, cursor_pos + image_size);
          handle_item_interaction(file_info, original_index);
        }
      }
      else {
        update_sorted_cache();
        auto& sorted_files = search_state.sorted_cache;

        for (const auto& [file_info, original_index] : sorted_files) {
          handle_keyboard_navigation(file_info.filename, pressed_key);

          table_next_row();
          table_set_column_index(0);

          fan::vec2 row_pos = get_cursor_screen_pos();
          f32_t row_height = get_text_line_height_with_spacing();
          fan::vec2 cursor_pos = fan::vec2(get_window_pos()) + fan::vec2(get_cursor_pos()) +
            fan::vec2(get_scroll_x(), -get_scroll_y());
          fan::vec2 image_size = fan::vec2(thumbnail_size / 4, thumbnail_size / 4);

          std::string space;
          while (calc_text_size(space.c_str()).x < image_size.x) space += " ";
          std::string str = space + file_info.filename;

          bool item_clicked = selectable(std::string_view(str), directory_cache[original_index].is_selected, selectable_flags_span_all_columns);

          if (item_clicked) {
            io_t& io = get_io();
            if (io.KeyCtrl) {
              directory_cache[original_index].is_selected = !directory_cache[original_index].is_selected;
            }
            else {
              for (auto& f : directory_cache) f.is_selected = false;
              directory_cache[original_index].is_selected = true;
            }
          }

          if (selection_state.is_selecting) {
            fan::vec2 rect_min = fan::vec2(
              std::min(selection_state.selection_start.x, selection_state.selection_end.x),
              std::min(selection_state.selection_start.y, selection_state.selection_end.y)
            );
            fan::vec2 rect_max = fan::vec2(
              std::max(selection_state.selection_start.x, selection_state.selection_end.x),
              std::max(selection_state.selection_start.y, selection_state.selection_end.y)
            );
            fan::vec2 row_min = row_pos;
            fan::vec2 row_max = fan::vec2(row_pos.x + get_content_region_avail().x, row_pos.y + row_height);
            bool overlaps = !(rect_max.x < row_min.x || rect_min.x > row_max.x ||
              rect_max.y < row_min.y || rect_min.y > row_max.y);
            if (overlaps) directory_cache[original_index].is_selected = true;
            else if (!selection_state.ctrl_held) directory_cache[original_index].is_selected = false;
          }

          handle_right_click(file_info.filename);

          texture_id_t texture_id = (texture_id_t)fan::graphics::image_get_handle(
            fan::graphics::is_image_valid(file_info.preview_image) ? file_info.preview_image
            : file_info.is_directory ? icon_directory : icon_file
          );
          get_window_draw_list()->AddImage(texture_id, cursor_pos, cursor_pos + image_size);
          handle_item_interaction(file_info, original_index);
        }
      }
      end_table();
    }
  }

#if defined(FAN_2D)
  void fragment_shader_editor(uint16_t shape_type, std::string* fragment, bool* shader_compiled) {
    auto& shader = fan::graphics::shader_get_data(shape_type);
    if (fragment->empty()) {
      *fragment = shader.sfragment;
    }
    if (begin("shader editor", 0, window_flags_no_saved_settings)) {
      if (!*shader_compiled) {
        gui::text("Failed to compile shader", fan::colors::red);
      }

      if (gui::input_text_multiline("##Shader Code", fragment, gui::get_content_region_avail(), gui::input_text_flags_allow_tab_input)) {
        *shader_compiled = fan::graphics::shader_update_fragment(shape_type, shader.path_fragment, *fragment);
      }
      end();
    }
  }
#endif

  // called inside window begin end
  void animated_popup_window(
    std::string_view popup_id,
    const fan::vec2& popup_size,
    const fan::vec2& start_pos,
    const fan::vec2& target_pos,
    bool trigger_popup,
    const std::function<void()>& content_cb,
    f32_t anim_duration,
    f32_t hide_delay
  ) {
    std::string popup_id_str = std::string(popup_id);
    std::string id_anim_time = popup_id_str + "_anim_time";
    std::string id_hide_timer = popup_id_str + "_hide_timer";
    std::string id_hovering = popup_id_str + "_hovering";
    std::string id_visible = popup_id_str + "_visible";

    id_t anim_time_id = get_id(std::string_view(id_anim_time));
    id_t hide_timer_id = get_id(std::string_view(id_hide_timer));
    id_t hovering_popup_id = get_id(std::string_view(id_hovering));
    id_t popup_visible_id = get_id(std::string_view(id_visible));

    f32_t delta_time = get_io().DeltaTime;

    storage_t* storage = get_state_storage();
    f32_t popup_anim_time = storage->GetFloat(anim_time_id, 0.0f);
    f32_t hide_timer = storage->GetFloat(hide_timer_id, 0.0f);
    bool hovering_popup = storage->GetBool(hovering_popup_id, false);
    bool popup_visible = storage->GetBool(popup_visible_id, false);

    if (trigger_popup || hovering_popup) {
      popup_visible = true;
      hide_timer = 0.0f;
    }
    else {
      hide_timer += delta_time;
    }

    if (popup_visible) {
      popup_anim_time = hide_timer < hide_delay
        ? std::min(popup_anim_time + delta_time, anim_duration)
        : std::max(popup_anim_time - delta_time, 0.0f);

      if (popup_anim_time == 0.0f) {
        popup_visible = false;
      }

      if (popup_anim_time > 0.0f) {
        f32_t t = popup_anim_time / anim_duration;
        t = t * t * (3.0f - 2.0f * t); // smoothstep

        fan::vec2 popup_pos = start_pos + (target_pos - start_pos) * t;

        set_next_window_pos(popup_pos);
        set_next_window_size(popup_size);

        if (begin(popup_id, nullptr,
          window_flags_no_title_bar | window_flags_no_resize |
          window_flags_no_move | window_flags_no_saved_settings |
          window_flags_no_collapse | window_flags_no_scrollbar |
          window_flags_no_focus_on_appearing)) {

          hovering_popup = is_window_hovered(
            hovered_flags_child_windows |
            hovered_flags_no_popup_hierarchy |
            hovered_flags_allow_when_blocked_by_popup |
            hovered_flags_allow_when_blocked_by_active_item
          );

          content_cb();
        }
        end();
      }
    }

    storage->SetFloat(anim_time_id, popup_anim_time);
    storage->SetFloat(hide_timer_id, hide_timer);
    storage->SetBool(hovering_popup_id, hovering_popup);
    storage->SetBool(popup_visible_id, popup_visible);
  }

#if defined(FAN_2D)

  bool sprite_animations_t::render_list_box(fan::graphics::sprite_sheet_id_t& shape_sprite_sheet_id) {
    bool list_item_changed = false;

    gui::begin_child("animations_tool_bar", 0, 1);
    gui::set_cursor_pos_y(animation_names_padding);
    gui::indent(animation_names_padding);

    if (gui::button("+")) {
      fan::graphics::sprite_sheet_t animation;
      animation.name = std::to_string((uint32_t)fan::graphics::ss_counter()); // think this over
      shape_sprite_sheet_id = fan::graphics::add_shape_sprite_sheet(shape_sprite_sheet_id, animation);
    }
    if (!shape_sprite_sheet_id) {
      gui::unindent(animation_names_padding);
      gui::end_child();
      return false;
    }
    gui::push_item_width(gui::get_window_size().x * 0.8f);
    for (auto [i, animation_nr] : fan::enumerate(fan::graphics::get_shape_sprite_sheets(shape_sprite_sheet_id))) {
      auto& animation = fan::graphics::get_sprite_sheet(animation_nr);
      if (animation.name == animation_list_name_to_edit) {
        std::snprintf(animation_list_name_edit_buffer.data(), animation_list_name_edit_buffer.size() + 1, "%s", animation.name.c_str());
        gui::push_id(i);
        if (set_focus) {
          gui::set_keyboard_focus_here();
          set_focus = false;
        }

        if (gui::input_text("##edit", &animation_list_name_edit_buffer, gui::input_text_flags_enter_returns_true)) {
          if (animation_list_name_edit_buffer != animation.name) {
            fan::graphics::rename_shape_sprite_sheet(shape_sprite_sheet_id, animation.name, animation_list_name_edit_buffer);
            animation.name = animation_list_name_edit_buffer;
            animation_list_name_to_edit.clear();
            gui::pop_id();
            break;
          }
          else {
            animation_list_name_to_edit.clear();
            gui::pop_id();
            break;
          }
        }
        gui::pop_id();
      }
      else {
        gui::push_id(i);
        if (gui::selectable(std::string_view(animation.name), current_animation_nr && current_animation_nr == animation_nr, gui::selectable_flags_allow_double_click, fan::vec2(gui::get_content_region_avail().x * 0.8f, 0))) {
          if (gui::is_mouse_double_clicked()) {
            animation_list_name_to_edit = animation.name;
            set_focus = true;
          }
          current_animation_shape_nr = shape_sprite_sheet_id;
          current_animation_nr = animation_nr;
          list_item_changed = true;
        }
        gui::pop_id();
      }
    }
    gui::pop_item_width();
    gui::unindent(animation_names_padding);
    gui::end_child();
    return list_item_changed;
  }

  bool sprite_animations_t::render_selectable_frames(fan::graphics::sprite_sheet_t& current_sprite_sheet) {
    bool changed = false;
    if (fan::window::is_mouse_released()) {
      previous_hold_selected.clear();
    }
    int grid_index = 0;
    bool first_button = true;

    for (int i = 0; i < current_sprite_sheet.images.size(); ++i) {
      auto current_image = current_sprite_sheet.images[i];
      int hframes = current_image.hframes;
      int vframes = current_image.vframes;
      for (int y = 0; y < vframes; ++y) {
        for (int x = 0; x < hframes; ++x) {
          fan::vec2 tc_size = fan::vec2(1.0 / hframes, 1.0 / vframes);
          fan::vec2 uv_src = fan::vec2(
            tc_size.x * x,
            tc_size.y * y
          );
          fan::vec2 uv_dst = uv_src + fan::vec2(1.0 / hframes, 1.0 / vframes);
          gui::push_id(grid_index);

          f32_t button_width = 128 + gui::get_style().ItemSpacing.x;
          f32_t window_width = gui::get_content_region_avail().x;
          f32_t current_line_width = gui::get_cursor_pos().x - gui::get_window_pos().x;

          if (current_line_width + button_width > window_width && !first_button) {
            gui::new_line();
          }

          fan::vec2 cursor_screen_pos = gui::get_cursor_screen_pos() + fan::vec2(7.f, 0);
          gui::image_button("", current_image.image, 128, uv_src, uv_dst);
          auto& sf = current_sprite_sheet.selected_frames;
          auto it_found = std::find_if(sf.begin(), sf.end(), [a = grid_index](int b) {
            return a == b;
          });
          bool is_found = it_found != sf.end();
          auto previous_hold_it = std::find(previous_hold_selected.begin(), previous_hold_selected.end(), grid_index);
          bool was_added_by_hold_before = previous_hold_it != previous_hold_selected.end();
          if (gui::is_item_held() && !was_added_by_hold_before) {
            if (is_found == false) {
              sf.push_back(grid_index);
              previous_hold_selected.push_back(grid_index);
              changed = true;
            }
            else {
              changed = true;
              it_found = sf.erase(it_found);
              is_found = false;
              previous_hold_selected.push_back(grid_index);
            }
          }
          if (is_found) {
            gui::text_outlined({ .pos = cursor_screen_pos }, std::distance(sf.begin(), it_found));
            gui::text({ .pos = cursor_screen_pos }, std::distance(sf.begin(), it_found));
          }
          gui::pop_id();

          if (!(y == vframes - 1 && x == hframes - 1 && i == current_sprite_sheet.images.size() - 1)) {
            gui::same_line();
          }

          first_button = false;
          ++grid_index;
        }
      }
    }
    return changed;
  }

  bool sprite_animations_t::render(std::string_view drag_drop_id, fan::graphics::sprite_sheet_id_t& shape_sprite_sheet_id) {
    gui::push_style_var(gui::style_var_item_spacing, fan::vec2(12.f, 12.f));
    gui::columns(2, "animation_columns", false);
    gui::set_column_width(0, gui::get_window_size().x * 0.2f);

    bool list_changed = render_list_box(shape_sprite_sheet_id);

    gui::next_column();

    gui::begin_child("animation_window_right", 0, 1, gui::window_flags_horizontal_scrollbar);

    gui::push_item_width(72);
    gui::indent(animation_names_padding);
    gui::set_cursor_pos_y(animation_names_padding);
    toggle_play_animation = false;
    if (gui::image_button("play_button", fan::graphics::icons.play, 32)) {
      play_animation = true;
      toggle_play_animation = true;
    }
    gui::same_line();
    if (gui::image_button("pause_button", fan::graphics::icons.pause, 32)) {
      play_animation = false;
      toggle_play_animation = true;
    }
    typename fan::graphics::ss_map_t::iterator current_sprite_sheet;
    if (!current_animation_nr) {
      goto g_end_frame;
    }
    current_sprite_sheet = fan::graphics::all_sprite_sheets().find(current_animation_nr);
    if (current_sprite_sheet == fan::graphics::all_sprite_sheets().end()) {
    g_end_frame:
      gui::columns(1);
      gui::end_child();
      gui::pop_style_var();
      return list_changed;
    }

    gui::same_line(0, 20.f);

    gui::slider_flags_t slider_flags = slider_flags_always_clamp | gui::slider_flags_no_speed_tweaks;
    list_changed |= gui::drag("fps", &current_sprite_sheet->second.fps, 1, 0, 244, slider_flags);
    if (gui::button("add sprite sheet")) {
      adding_sprite_sheet = true;
    }
    if (adding_sprite_sheet && gui::begin("add_animations_sprite_sheet")) {
      gui::text_box("Drop sprite sheet here", fan::vec2(256, 64));
      gui::receive_drag_drop_target(drag_drop_id, [this](const std::string& file_path) {
        if (fan::image::valid(file_path)) {
          sprite_sheet_drag_drop_name = file_path;
        }
        else {
          fan::graphics::gui::print("Warning: drop target not valid (requires image file)");
        }
      });

      gui::drag("Horizontal frames", &hframes, 1, 0, 1024, slider_flags);
      gui::drag("Vertical frames", &vframes, 1, 0, 1024, slider_flags);

      if (!sprite_sheet_drag_drop_name.empty()) {
        gui::separator();
        gui::text("Preview:");

        fan::graphics::image_t preview_image = fan::graphics::image_load(sprite_sheet_drag_drop_name);

        if (fan::graphics::is_image_valid(preview_image)) {
          f32_t content_width = hframes * (64 + gui::get_style().ItemSpacing.x);
          f32_t content_height = vframes * (64 + gui::get_style().ItemSpacing.y);

          if (gui::begin_child("sprite_preview", fan::vec2(0, std::min(content_height + 20, 300.0f)), true, window_flags_horizontal_scrollbar)) {
            for (int y = 0; y < vframes; ++y) {
              for (int x = 0; x < hframes; ++x) {
                fan::vec2 tc_size = fan::vec2(1.0 / hframes, 1.0 / vframes);
                fan::vec2 uv_src = fan::vec2(tc_size.x * x, tc_size.y * y);
                fan::vec2 uv_dst = uv_src + tc_size;

                gui::push_id(y * hframes + x);
                gui::image_button("", preview_image, 64, uv_src, uv_dst);
                gui::pop_id();

                if (x != hframes - 1) {
                  gui::same_line();
                }
              }
            }
          }
          gui::end_child();
        }
      }

      if (gui::button("Add")) {
        if (auto it = fan::graphics::all_sprite_sheets().find(current_animation_nr); it != fan::graphics::all_sprite_sheets().end()) {
          auto& anim = it->second;
          fan::graphics::sprite_sheet_t::image_t new_image;
          new_image.image = fan::graphics::image_load(sprite_sheet_drag_drop_name);
          new_image.hframes = hframes;
          new_image.vframes = vframes;
          anim.images.push_back(new_image);
          sprite_sheet_drag_drop_name.clear();
        }
        adding_sprite_sheet = false;
        list_changed |= 1;
      }

      gui::end();
    }

    gui::separator();

    fan::vec2 cursor_pos = gui::get_cursor_pos();
    if (!drag_drop_id.empty()) {
      fan::vec2 child_size = gui::get_window_size();
      dummy(child_size);
      gui::receive_drag_drop_target(drag_drop_id, [this](const std::string& file_paths) {
        for (const std::string& file_path : fan::split(file_paths, ";")) {
          if (fan::image::valid(file_path)) {
            if (auto it = fan::graphics::all_sprite_sheets().find(current_animation_nr); it != fan::graphics::all_sprite_sheets().end()) {
              auto& anim = it->second;
              fan::graphics::sprite_sheet_t::image_t new_image;
              new_image.image = fan::graphics::image_load(file_path);
              anim.images.push_back(new_image);
            }
          }
          else {
            fan::graphics::gui::print("Warning: drop target not valid (requires image file)");
          }
        }
      });
    }
    gui::set_cursor_pos(cursor_pos);

    list_changed |= render_selectable_frames(current_sprite_sheet->second);

    gui::unindent(animation_names_padding);
    gui::end_child();
    gui::columns(1);
    gui::pop_style_var();
    return list_changed;
  }

  particle_editor_t::particle_editor_t() {
    set_particle_shape(std::move(particle_shape));
  }

  fan::graphics::shapes::particles_t::ri_t& particle_editor_t::get_ri() {
    return *(fan::graphics::shapes::particles_t::ri_t*)particle_shape.GetData(fan::graphics::g_shapes->shaper);
  }

  void particle_editor_t::render_menu() {
    if (begin_main_menu_bar()) {
      if (begin_menu("File")) {
        if (menu_item("Open..", "Ctrl+O")) {
          fan::graphics::open_file("json;fmm", [&](std::string_view path) {
            filename = path;
            particle_shape = shape_from_json(filename);
            particle_image_sprite.set_image(particle_shape.get_image());
          });
        }
        if (menu_item("Save as", "Ctrl+Shift+S")) {
          fan::graphics::save_file("json;fmm", [&](std::string_view path) {
            filename = path;
            fout(filename);
          });
        }
        end_menu();
      }
      end_main_menu_bar();
    }
  }

  void particle_editor_t::render_settings() {
    color_edit4("background color", &bg_color);
    gui::render_texture_property(particle_image_sprite, 0, "Particle texture");
    render_image_filter_property(particle_image_sprite, "Particle texture image filter");
    particle_shape.set_image(particle_image_sprite.get_image());
    shape_properties(particle_shape);

    if (fan::window::is_key_clicked(fan::key_s) && fan::window::is_key_down(fan::key_left_control)) {
      fout(filename);
    }
  }

  void particle_editor_t::render() {
    render_menu();
    render_settings();
  }

  void particle_editor_t::fout(std::string_view f) {
    filename = f;
    fan::json json_data;
    fan::graphics::shape_to_json(particle_shape, &json_data);
    if (!filename.ends_with(".json")) {
      filename += ".json";
    }

    // set relative path from json to image
    json_data.find_and_iterate("image_path", [this](fan::json& value) {
      value = fan::io::file::relative_path(value.get<std::string>(), filename).generic_string();
    });

    fan::graphics::gui::print_success("File saved to " + std::filesystem::absolute(filename).generic_string());
    fan::io::file::write(filename, json_data.dump(2), std::ios_base::binary);
  }

  void particle_editor_t::set_particle_shape(fan::graphics::shape_t&& shape) {
    particle_shape = std::move(shape);
    g_shapes->visit_shape_draw_data(particle_shape.NRI, [&]<typename T>(T & properties) {
      if constexpr (requires{ properties.image; }) {
        particle_image_sprite = fan::graphics::shape_t(
          fan::graphics::shapes::sprite_t::properties_t {
            .size = 0,
            .image = properties.image
          },
          false/*culling*/
        );
      }
    });
  }

#endif

  dialogue_box_t::text_delayed_t::~text_delayed_t() {
    dialogue_line_finished = true;
    character_advance_task = {};
  }

  void dialogue_box_t::text_delayed_t::render(dialogue_box_t* This, dialogue_box_t::drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {

    if (dialogue_line_finished == false && !character_advance_task.owner) {
      character_advance_task = [This, nr]() -> fan::event::task_t {
        text_delayed_t* text_delayed = nullptr;

        for (auto& node : This->drawables) {
          if (node.id == nr) {
            text_delayed = dynamic_cast<text_delayed_t*>(node.ptr.get());
            break;
          }
        }

        if (text_delayed == nullptr) {
          co_return;
        }

        while (text_delayed->render_pos < text_delayed->text.size() && !text_delayed->dialogue_line_finished && text_delayed->character_per_s) {
          ++text_delayed->render_pos;
          co_await fan::co_sleep(1000 / text_delayed->character_per_s);
        }
      }();
    }

    text_partial_render(text, render_pos, wrap_width, line_spacing);

    if (render_pos == text.size()) {
      dialogue_line_finished = true;
    }

    if (dialogue_line_finished && blink_timer.finished()) {
      if (render_cursor) {
        text.push_back('|');
        render_pos = text.size();
      }
      else {
        if (text.back() == '|') {
          text.pop_back();
          render_pos = text.size();
        }
      }
      render_cursor = !render_cursor;
      blink_timer.restart();
    }
  }

  void dialogue_box_t::text_t::render(dialogue_box_t* This, dialogue_box_t::drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
    gui::text(text);
  }

  void dialogue_box_t::button_t::render(dialogue_box_t* This, dialogue_box_t::drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
    if (This->wait_user) {
      fan::vec2 button_size = 0;
      fan::vec2 text_size = gui::calc_text_size(text);
      f32_t padding_x = get_style().FramePadding.x;
      f32_t padding_y = get_style().FramePadding.y;
      button_size = fan::vec2(text_size.x + padding_x * 2.0f, text_size.y + padding_y * 2.0f);

      fan::vec2 cursor = (position * get_window_size()) - (size == 0 ? button_size / 2 : size / 2);
      cursor.x += get_style().WindowPadding.x;
      cursor.y += get_style().WindowPadding.y;
      set_cursor_pos(cursor);

      if (gui::button(std::string_view(text), size == 0 ? button_size : size)) {
        This->button_choice = nr;
        This->wait_user = false;
      }
    }
  }

  void dialogue_box_t::separator_t::render(dialogue_box_t* This, dialogue_box_t::drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
    seperator();
  }

  dialogue_box_t::dialogue_box_t() {
    fan::window::add_input_action(fan::mouse_left, "skip or continue dialog");
  }

  // 0-1
  void dialogue_box_t::set_cursor_position(const fan::vec2& pos) {
    this->cursor_position = pos;
  }

  void dialogue_box_t::set_indent(f32_t indent) {
    this->indent = indent;
  }

  fan::event::runv_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::text_delayed(
    std::string_view character_name,
    std::string_view text,
    int characters_per_second) {

    auto node_id = next_id++;

    auto td = std::make_unique<text_delayed_t>();
    td->character_per_s = characters_per_second;
    td->text = text;
    td->render_pos = 0;
    td->dialogue_line_finished = false;

    drawables.push_back({node_id, std::move(td)});

    co_return node_id;
  }

  fan::event::runv_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::text(const std::string& text) {
    auto node_id = next_id++;

    auto text_drawable = std::make_unique<text_t>();
    text_drawable->text = text;

    drawables.push_back({node_id, std::move(text_drawable)});
    co_return node_id;
  }

  fan::event::runv_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::button(const std::string& text, const fan::vec2& position, const fan::vec2& size) {
    auto node_id = next_id++;

    auto btn = std::make_unique<button_t>();
    btn->position = position;
    btn->size = size;
    btn->text = text;

    drawables.push_back({node_id, std::move(btn)});

    co_return node_id;
  }

  // default width 80% of the window
  fan::event::runv_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::separator(f32_t width) {
    auto node_id = next_id++;

    auto sep = std::make_unique<separator_t>();
    // sep->width = width; ?
    drawables.push_back({node_id, std::move(sep)});
    co_return node_id;
  }

  int dialogue_box_t::get_button_choice() {
    return button_choice;
  }

  fan::event::task_t dialogue_box_t::wait_user_input() {
    wait_user = true;
    while (wait_user) {
      co_await fan::co_sleep(10);
    }
  }

  void dialogue_box_t::render(fan::str_view_t window_name, font_t* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing, const std::function<void()>& inside_window_cb) {
    push_font(font);

    fan::vec2 root_window_size = get_window_size();
    fan::vec2 next_window_pos;
    next_window_pos.x = (root_window_size.x - window_size.x) / 2.0f;
    next_window_pos.y = (root_window_size.y - window_size.y) / 1.1;
    set_next_window_pos(next_window_pos);

    set_next_window_size(window_size);
    begin(window_name, nullptr,
      window_flags_no_nav_inputs | window_flags_no_title_bar |
      window_flags_no_move | window_flags_no_resize | window_flags_no_scrollbar
    );

    f32_t current_font_size = get_font()->FontSize;
    f32_t scale_factor = font_size / current_font_size;
    set_window_font_scale(scale_factor);

    // begin_child((fan::random::string(10) + "child").c_str(), fan::vec2(wrap_width, 0), 0, window_flags_no_nav_inputs | window_flags_no_title_bar |
    //   window_flags_no_move | window_flags_no_resize | window_flags_no_scrollbar | window_flags_no_background);

    inside_window_cb();

    render_content_cb(cursor_position == -1 ? fan::vec2(get_style().WindowPadding) : cursor_position, indent);

    for (auto& drawable : drawables) {
      drawable.ptr->render(this, drawable.id, window_size, wrap_width, line_spacing);
    }
    set_window_font_scale(1.0f);

    bool has_buttons = false;
    for (auto& drawable : drawables) {
      if (dynamic_cast<button_t*>(drawable.ptr.get()) != nullptr) {
        has_buttons = true;
        break;
      }
    }

    bool dialogue_line_finished = fan::window::is_input_action_active("skip or continue dialog") && 
      is_window_hovered(hovered_flags_child_windows | hovered_flags_allow_when_blocked_by_popup | hovered_flags_allow_when_blocked_by_active_item) &&
      !has_buttons;

    if (dialogue_line_finished) {
      wait_user = false;
      clear();
    }

    end();
    pop_font();

  }

  fan::event::runv_t<int> dialogue_box_t::choice(
    std::string_view character_name,
    std::string_view question_text,
    std::span<const std::string_view> options,
    const fan::vec2& start,
    f32_t y_step) {
    button_choice = -1;

    auto text_id = co_await text_delayed(character_name, question_text);

    std::vector<drawable_nr_t> ids;
    ids.reserve(options.size());

    for (size_t i = 0; i < options.size(); ++i) {
      fan::vec2 pos = start;
      pos.y += i * y_step;
      ids.push_back(co_await button(std::string(options[i]), pos));
    }

    co_await wait_user_input();

    int result = -1;
    auto it = std::find(ids.begin(), ids.end(), button_choice);
    if (it != ids.end()) {
      result = std::distance(ids.begin(), it);
    }

    ids.push_back(text_id);
    std::erase_if(drawables, [&](const drawable_node_t& node) {
      return std::find(ids.begin(), ids.end(), node.id) != ids.end();
    });

    wait_user = false;

    co_return result;
  }

  void dialogue_box_t::clear() {
    drawables.clear();
  }

  void dialogue_box_t::default_render_content(const fan::vec2& cursor_pos, f32_t indent) {
    set_cursor_pos(cursor_pos);
    gui::indent(indent);
  }

  void text_partial_render(const std::string& text, size_t render_pos, f32_t wrap_width, f32_t line_spacing) {
    static auto find_next_word = [](const std::string& str, std::size_t offset) -> std::size_t {
      std::size_t found = str.find(' ', offset);
      if (found == std::string::npos) {
        found = str.size();
      }
      if (found != std::string::npos) {
      }
      return found;
    };
    static auto find_previous_word = [](const std::string& str, std::size_t offset) -> std::size_t {
      std::size_t found = str.rfind(' ', offset);
      if (found == std::string::npos) {
        found = str.size();
      }
      if (found != std::string::npos) {
      }
      return found;
    };

    std::vector<std::string> lines;
    std::size_t previous_word = 0;
    std::size_t previous_push = 0;
    bool found = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
      std::size_t word_index = text.find(' ', i);
      if (word_index == std::string::npos) {
        word_index = text.size();
      }

      std::string str = text.substr(previous_push, word_index - previous_push);
      f32_t width = calc_text_size(str).x;

      if (width >= wrap_width) {
        if (previous_push == previous_word) {
          lines.push_back(text.substr(previous_push, i - previous_push));
          previous_push = i;
        }
        else {
          lines.push_back(text.substr(previous_push, previous_word - previous_push));
          previous_push = previous_word + 1;
          i = previous_word;
        }
      }
      previous_word = word_index;
      i = word_index;
    }

    if (previous_push < text.size()) {
      lines.push_back(text.substr(previous_push));
    }

    std::size_t empty_lines = 0;
    std::size_t character_offset = 0;
    fan::vec2 pos = get_cursor_screen_pos();
    for (const auto& line : lines) {
      if (line.empty()) {
        ++empty_lines;
        continue;
      }
      std::size_t empty = 0;
      if (empty >= line.size()) {
        break;
      }
      while (line[empty] == ' ') {
        if (empty >= line.size()) {
          break;
        }
        ++empty;
      }
      if (character_offset >= render_pos) {
        break;
      }
      std::string render_text = line.substr(empty).c_str();
      set_cursor_screen_pos(pos);
      if (character_offset + render_text.size() >= render_pos) {
        gui::text(render_text.substr(0, render_pos - character_offset));
        break;
      }
      else {
        gui::text(render_text);
        if (render_text.back() != ' ') {
          character_offset += 1;
        }
        character_offset += render_text.size();
        pos.y += get_line_height_with_spacing() + line_spacing;
      }
    }
    if (empty_lines) {
      gui::text(fan::colors::red, "warning empty lines:", empty_lines);
    }
  }

  void render_texture_property(
    fan::graphics::shape_t& shape,
    int index,
    fan::str_view_t label,
    const std::string& asset_path,
    f32_t image_size,
    const char* receive_drag_drop_target_name
  ) {
    using namespace fan::graphics;
    auto current_image = index > 0 ? shape.get_images()[index - 1] : shape.get_image();
    if (current_image.iic()) {
      current_image = fan::graphics::ctx().default_texture;
    }
    fan::vec2 uv0 = shape.get_tc_position(), uv1 = shape.get_tc_size();
    uv1 += uv0;
    gui::image(current_image, fan::vec2(image_size), uv0, uv1);
    gui::receive_drag_drop_target(receive_drag_drop_target_name, [&, asset_path, index](const std::string& path) {
      auto new_image = fan::graphics::image_load((std::filesystem::path(asset_path) / path).generic_string());
      if (index > 0) {
        auto images = shape.get_images();
        images[index - 1] = new_image;
        shape.set_images(images);
      }
      else {
        shape.set_image(new_image);
      }
      shape.set_tc_position(0);
      shape.set_tc_size(1);
    });
    gui::same_line();
    gui::text(label);
  }
  void render_image_filter_property(
    fan::graphics::shape_t& shape,
    fan::str_view_t label
  ) {
    using namespace fan::graphics;

    auto current_image = shape.get_image();
    int current_filter = fan::graphics::image_get_settings(current_image).min_filter;

    static const char* image_filters[] = {"nearest", "linear"};

    if (gui::combo(label, &current_filter, image_filters, std::size(image_filters))) {
      image_load_properties_t ilp;
      ilp.min_filter = current_filter;
      ilp.mag_filter = current_filter;

      fan::graphics::image_set_settings(shape.get_image(), ilp);

      auto images = shape.get_images();
      for (size_t i = 0; i < images.size(); ++i) {
        if (!images[i].iic()) {
          fan::graphics::image_set_settings(images[i], ilp);
        }
      }
    }
  }

  void shader_controls(fan::graphics::shader_t shader_id, const shader_contols_t& controls) {
    static std::unordered_map<
      std::remove_cvref_t<decltype(shader_id.gint())>,
      std::vector<std::array<uint8_t, sizeof(fan::vec4)>>
    > map;

    auto& shader_list = *fan::graphics::ctx().shader_list;
    auto& shader_data = shader_list[shader_id];
    auto& table = shader_data.uniform_type_table;

    auto [it, inserted] = map.try_emplace(shader_id.gint());
    auto& table_data = it->second;

    if (inserted) {
      table_data.resize(table.size());
      uint32_t table_idx = 0;

      #define create_get_case(shader_var_type, type) \
        case fan::get_hash(std::string_view(shader_var_type)): { \
          fan::graphics::shader_get_value(shader_id, var.first, *(type*)var_data); \
          break; \
        }

      for (auto& var : table) {
        uint8_t* var_data = table_data[table_idx++].data();
        switch (fan::get_hash(var.second)) {
          create_get_case("bool",  bool)
          create_get_case("int",   int)
          create_get_case("uint",  uint32_t)
          create_get_case("float", f32_t)
          create_get_case("vec2",  fan::vec2)
          create_get_case("vec3",  fan::vec3)
          create_get_case("vec4",  fan::vec4)
        }
      }
      #undef create_get_case
    }

    #define create_case(shader_var_type, type, gui_expr) \
      case fan::get_hash(std::string_view(shader_var_type)): { \
        if (gui_expr) { \
          fan::graphics::shader_set_value(shader_id, var.first, *(type*)var_data); \
        } \
        break; \
      }

    uint32_t table_idx = 0;
    std::string_view name = shader_data.path_fragment;
    gui::begin(name.empty() ? "##" : name);

    for (auto& var : table) {
      uint8_t* var_data = table_data[table_idx++].data();
      switch (fan::get_hash(var.second)) {
        create_case("bool",  bool,      gui::checkbox(var.first, (bool*)var_data))
        create_case("int",   int,       gui::drag(var.first, (int*)var_data))
        create_case("uint",  uint32_t,  gui::drag(var.first, (uint32_t*)var_data))
        create_case("float", f32_t,     gui::drag(var.first, (f32_t*)var_data))
        create_case("vec2",  fan::vec2, gui::drag(var.first, (fan::vec2*)var_data))
        create_case("vec3",  fan::vec3, controls.vec3_as_color ? gui::color_edit3(var.first, (fan::vec3*)var_data) : gui::drag(var.first, (fan::vec3*)var_data))
        case fan::get_hash(std::string_view("vec4")): {
          if (controls.vec4_as_color) {
            if (gui::color_edit4(var.first, (fan::color*)var_data)) {
              fan::graphics::shader_set_value(shader_id, var.first, *(fan::color*)var_data);
            }
          } else {
            if (gui::drag(var.first, (fan::vec4*)var_data)) {
              fan::graphics::shader_set_value(shader_id, var.first, *(fan::vec4*)var_data);
            }
          }
          break;
        }
      }
    }
    gui::end();

    #undef create_case
  }

  void hex_editor_t::render(fan::io::data_provider_t& data) {
    render("", data);
  }

  bool hex_editor_t::has_selection() const {
    return sel_start.has_value();
  }

  std::pair<uint64_t, uint64_t> hex_editor_t::get_selection_bounds() const {
    return std::minmax(sel_start.value(), sel_end.value());
  }

  bool hex_editor_t::is_selected(uint64_t idx) const {
    if (!has_selection()) return false;
    auto [lo, hi] = get_selection_bounds();
    return idx >= lo && idx <= hi;
  }

  void hex_editor_t::update_selection(uint64_t idx, bool cell_hovered) {
    if (!cell_hovered) return;

    if (ImGui::IsMouseClicked(0) && !ImGui::IsMouseDragging(0, 3.0f)) {
      sel_start = idx;
      sel_end = idx;
    } else if (ImGui::IsMouseDragging(0, 3.0f) && has_selection()) {
      sel_end = idx;
    }
  }

  uint32_t hex_editor_t::get_cell_flags(bool is_dragging, bool is_hex) const {
    uint32_t flags = is_hex ? gui::input_text_flags_chars_hexadecimal | gui::input_text_flags_chars_uppercase : 0;
    flags |= gui::input_text_flags_auto_select_all;
    flags |= gui::input_text_flags_no_horizontal_scroll;
    return flags;
  }

  f32_t hex_editor_t::get_spacing(uint64_t idx, uint64_t row_end, bool is_hex) const {
    if (!is_hex) return metrics.char_w * config.spacing_ascii_mult;
    return (idx + 1) % config.group_size.x == 0
      ? (metrics.char_w * config.spacing_hex_group_mult)
      : (metrics.char_w * config.spacing_hex_item_mult);
  }

  void hex_editor_t::render_data_inspector(fan::io::data_provider_t& data, bool little_endian) {
    auto offset = get_active_cell(data);
    if (!offset || !little_endian) return;

    if (auto c = gui::child_window("data_inspector")) {
      std::vector<uint8_t> result;
      data.read_range_padded(*offset, sizeof(uint64_t), result);

      static std::optional<uint64_t> last_offset;
      static std::string bufs[10];

      if (last_offset != *offset || !ImGui::IsAnyItemActive()) {
        last_offset = *offset;
        auto* p = result.data();
        bufs[0] = std::to_string(*(uint8_t*)p); bufs[1] = std::to_string(*(int8_t*)p);
        bufs[2] = std::to_string(*(uint16_t*)p); bufs[3] = std::to_string(*(int16_t*)p);
        bufs[4] = std::to_string(*(uint32_t*)p); bufs[5] = std::to_string(*(int32_t*)p);
        bufs[6] = std::to_string(*(uint64_t*)p); bufs[7] = std::to_string(*(int64_t*)p);
        bufs[8] = fan::format_scientific(std::bit_cast<f32_t>(*(uint32_t*)p));
        bufs[9] = fan::format_scientific(std::bit_cast<f64_t>(*(uint64_t*)p));
      }

      auto invalidate = [&] { last_offset = std::nullopt; };

      if (auto tbl = gui::table("##data_inspector_table", 3)) {
        gui::table_setup_column("Type");
        gui::table_setup_column("Unsigned (+)");
        gui::table_setup_column("Signed (±)");
        gui::table_headers_row();

        table_row_edit("int8", bufs[0], bufs[1],
          [&] { fan::io::inspector_write(data, *offset, (uint8_t)std::stoull(bufs[0])); invalidate(); },
          [&] { fan::io::inspector_write(data, *offset, (int8_t)std::stoll(bufs[1])); invalidate(); });
        table_row_edit("int16", bufs[2], bufs[3],
          [&] { fan::io::inspector_write(data, *offset, (uint16_t)std::stoull(bufs[2])); invalidate(); },
          [&] { fan::io::inspector_write(data, *offset, (int16_t)std::stoll(bufs[3])); invalidate(); });
        table_row_edit("int32", bufs[4], bufs[5],
          [&] { fan::io::inspector_write(data, *offset, (uint32_t)std::stoull(bufs[4])); invalidate(); },
          [&] { fan::io::inspector_write(data, *offset, (int32_t)std::stoll(bufs[5])); invalidate(); });
      }

      if (auto tbl = gui::table("##float_table", 2)) {
        gui::table_setup_column("Type", gui::table_column_flags_width_fixed, 100.f);
        gui::table_setup_column("Value");
        gui::table_headers_row();

        table_row_edit("uint64", bufs[6], [&] { fan::io::inspector_write(data, *offset, (uint64_t)std::stoull(bufs[6])); invalidate(); });
        table_row_edit("int64", bufs[7], [&] { fan::io::inspector_write(data, *offset, (uint64_t)std::stoll(bufs[7])); invalidate(); });
        table_row_edit("float", bufs[8], [&] { fan::io::inspector_write(data, *offset, std::bit_cast<uint32_t>(std::stof(bufs[8]))); invalidate(); });
        table_row_edit("double", bufs[9], [&] { fan::io::inspector_write(data, *offset, std::bit_cast<uint64_t>(std::stod(bufs[9]))); invalidate(); });
      }
    }
  }

  void hex_editor_t::render_cell(fan::io::data_provider_t& data, uint64_t idx, f32_t w, f32_t pad, bool is_dragging, bool is_hex) {
    fan::color col = is_selected(idx) ? config.col_text_sel : fan::color::nibble(data.read(idx));
    gui::style_scope_t s;
    s.color(gui::col_text, col);

    auto do_render = [&]() {
      gui::push_id(static_cast<const char*>(static_cast<const void*>(&data)) + (is_hex ? idx : ascii_id_offset + idx));
      gui::set_next_item_width(w + pad);

      auto dl = ImGui::GetWindowDrawList();
      fan::vec2 rmin = gui::get_cursor_screen_pos();
      fan::vec2 rmax = rmin + fan::vec2(w + pad, gui::get_frame_height() - 1.f);
      
      bool is_active_cell = (active_idx == idx && active_panel == (is_hex ? active_panel_t::hex : active_panel_t::ascii));
      bool cell_hovered = false;

      if (is_selected(idx)) dl->AddRectFilled(ImVec2(rmin.x, rmin.y), ImVec2(rmax.x, rmax.y), ImGui::ColorConvertFloat4ToU32(ImVec4(config.col_bg_sel.r, config.col_bg_sel.g, config.col_bg_sel.b, config.col_bg_sel.a)));
      else if (is_active_cell) dl->AddRectFilled(ImVec2(rmin.x, rmin.y), ImVec2(rmax.x, rmax.y), ImGui::ColorConvertFloat4ToU32(ImVec4(0.4f, 0.4f, 0.1f, 0.7f)));
      else if (prev_hex_hover_idx == idx || prev_ascii_hover_idx == idx) dl->AddRectFilled(ImVec2(rmin.x, rmin.y), ImVec2(rmax.x, rmax.y), ImGui::ColorConvertFloat4ToU32(ImVec4(config.col_bg_hover.r, config.col_bg_hover.g, config.col_bg_hover.b, config.col_bg_hover.a)));

      std::optional<uint64_t>& focus_idx = is_hex ? pending_focus_hex : pending_focus_ascii;
      if (focus_idx == idx) {
        gui::set_keyboard_focus_here(0);
        focus_idx = std::nullopt;
        active_idx = idx;
        is_active_cell = true;
        active_panel = is_hex ? active_panel_t::hex : active_panel_t::ascii;
      }

      if (is_active_cell) {
        if (active_edit_initialized_idx != idx) {
          active_edit_buf = is_hex ? fan::to_hex(data.read(idx), 2) : std::string(1, (char)data.read(idx));
          active_edit_initialized_idx = idx;
        }

        bool changed = gui::input_text("##v", &active_edit_buf, get_cell_flags(is_dragging, is_hex));
        cell_hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
        
        bool active = gui::is_item_active();
        bool backspace = active && fan::window::is_key_clicked(fan::key_backspace);

        if (active) {
          if (fan::window::is_key_down(fan::key_left_control)) {
            gui::clear_active_id();
          } else if (backspace && idx > 0) {
            data.write(idx - 1, 0);
            focus_idx = idx - 1;
            gui::clear_active_id();
          } else if (fan::window::is_key_clicked(fan::key_left) && idx > 0) {
            focus_idx = idx - 1;
            gui::clear_active_id();
          } else if (fan::window::is_key_clicked(fan::key_right) && idx + 1 < metrics.size) {
            focus_idx = idx + 1;
            gui::clear_active_id();
          }
        }

        if (changed && !backspace) {
          if (is_hex) {
            if (active_edit_buf.size() > 2) active_edit_buf = active_edit_buf.substr(active_edit_buf.size() - 2);
            data.write(idx, fan::parse_hex_byte(active_edit_buf));
            if (active_edit_buf.size() >= 2 && idx + 1 < metrics.size) {
              focus_idx = idx + 1;
              gui::clear_active_id();
            }
          } else {
            if (!active_edit_buf.empty()) {
              data.write(idx, (uint8_t)active_edit_buf.back());
              if (idx + 1 < metrics.size) focus_idx = idx + 1;
              gui::clear_active_id();
            } else {
              data.write(idx, 0);
              gui::clear_active_id();
            }
          }
        }
      } else {
        gui::invisible_button("##cell", fan::vec2(w, gui::get_frame_height()));
        cell_hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
        
        if (gui::is_item_clicked(0)) {
          active_idx = idx;
          active_panel = is_hex ? active_panel_t::hex : active_panel_t::ascii;
          focus_idx = idx;
        }

        char display_buf[3];
        if (is_hex) {
          snprintf(display_buf, sizeof(display_buf), "%02X", data.read(idx));
        } else {
          uint8_t b = data.read(idx);
          display_buf[0] = (b >= 32 && b < 127) ? b : '.';
          display_buf[1] = '\0';
        }
        dl->AddText(ImVec2(rmin.x, rmin.y), ImGui::ColorConvertFloat4ToU32(ImVec4(col.r, col.g, col.b, col.a)), display_buf);
      }

      if (cell_hovered) {
        if (is_hex) hovered_hex_idx = idx;
        else hovered_ascii_idx = idx;
      }

      update_selection(idx, cell_hovered);
      gui::pop_id();
    };

    if (is_hex) {
      gui::style_scope_t hex_style;
      hex_style.color(gui::col_frame_bg, fan::color(0.f, 0.f, 0.f, 0.f));
      hex_style.color(gui::col_frame_bg_hovered, fan::color(0.f, 0.f, 0.f, 0.f));
      hex_style.color(gui::col_frame_bg_active, fan::color(0.f, 0.f, 0.f, 0.f));
      do_render();
    } else {
      auto invisible = gui::make_invisible_input_style();
      do_render();
    }
  }

void hex_editor_t::render(const std::string_view window_name, fan::io::data_provider_t& data) {
  char buf[64];
  uint64_t flags = window_name.empty() ? gui::window_flags_no_saved_settings : 0;
  if (window_name.empty()) snprintf(buf, sizeof(buf), "hex_editor##%p", this);
  else snprintf(buf, sizeof(buf), "%.*s", (int)window_name.size(), window_name.data());

  gui::set_next_window_size(fan::vec2(1280.f, 720.f), gui::cond_first_use_ever);

  auto window = gui::window(buf, nullptr, flags);
  gui::window_move_title_bar_only();
  if (!window) return;

  gui::set_window_font_scale(1.0f);
  f32_t base_font_size = gui::get_font_size();
  metrics.char_w = gui::calc_text_size("0").x;
  f32_t base_cell_w = gui::calc_text_size("FF").x + (config.inner_pad * 2.f);
  f32_t base_ascii_w = gui::calc_text_size("F").x + (config.inner_pad * 2.f);

  f32_t hex_area_w = config.cols * base_cell_w + config.cols * metrics.char_w * config.spacing_hex_item_mult;
  f32_t total_base_w = gui::calc_text_size("00000000").x + hex_area_w + config.cols * base_ascii_w
    + gui::get_style().CellPadding.x * 6.f + 5.f;

  f32_t auto_scale = 1.0f;

  gui::zoom_scope_t _(user_zoom, base_font_size, auto_scale, config.zoom_speed, 5.f);

  gui::font_t* crisp_font = gui::get_font(base_font_size * auto_scale * user_zoom, gui::font::mono);
  metrics.scale = crisp_font->FontSize / base_font_size;
  metrics.cell_w = base_cell_w * metrics.scale;
  metrics.ascii_w = base_ascii_w * metrics.scale;
  metrics.char_w *= metrics.scale;
  metrics.size = data.size();
  metrics.rows = (metrics.size + config.cols - 1) / config.cols;

  gui::push_font(crisp_font);
  gui::set_window_font_scale(1.0f);

  auto& st = gui::get_style();
  gui::style_scope_t scaled_style;
  f32_t ys = config.y_spacing_mult;
  scaled_style.var(gui::style_var_item_spacing, fan::vec2(0.f, st.ItemSpacing.y) * metrics.scale);
  scaled_style.var(gui::style_var_cell_padding, fan::vec2(st.CellPadding.x, st.CellPadding.y * ys) * metrics.scale);
  scaled_style.var(gui::style_var_frame_padding, fan::vec2(config.inner_pad, st.FramePadding.y * ys) * metrics.scale);
  scaled_style.var(gui::style_var_window_padding, fan::vec2(st.WindowPadding.x, st.WindowPadding.y) * metrics.scale);

  bool dragging = ImGui::IsMouseDown(0) && has_selection() && sel_start != sel_end;

  if (dragging && !is_dragging) gui::clear_active_id();
  is_dragging = dragging;

  prev_hex_hover_idx = hovered_hex_idx; hovered_hex_idx = std::nullopt;
  prev_ascii_hover_idx = hovered_ascii_idx; hovered_ascii_idx = std::nullopt;

  if (gui::begin_table("##outer_split", 2, gui::table_flags_no_borders_in_body)) {
    gui::table_setup_column("##hex", gui::table_column_flags_width_fixed, (total_base_w - 15.f) * metrics.scale);
    gui::table_setup_column("##inspector", gui::table_column_flags_width_stretch);
    gui::table_next_row();
    gui::table_next_column();
    {
      auto& st = gui::get_style();
      gui::style_scope_t scaled_style;
      f32_t ys = config.y_spacing_mult;
      scaled_style.var(gui::style_var_item_spacing, fan::vec2(0.f, st.ItemSpacing.y) * metrics.scale);
      scaled_style.var(gui::style_var_cell_padding, fan::vec2(st.CellPadding.x, st.CellPadding.y * ys) * metrics.scale);
      scaled_style.var(gui::style_var_frame_padding, fan::vec2(config.inner_pad, st.FramePadding.y * ys) * metrics.scale);
      scaled_style.var(gui::style_var_window_padding, fan::vec2(st.WindowPadding.x, st.WindowPadding.y) * metrics.scale);
      f32_t row_height = gui::get_text_line_height() + gui::get_style().ItemSpacing.y;
      uint64_t first_visible_row = static_cast<uint64_t>(std::max(0.0f, gui::get_scroll_y() / row_height));
      uint64_t visible_row_count = static_cast<uint64_t>(gui::get_window_size().y / row_height) + 2;

      gui::dummy(fan::vec2(0, std::min<uint64_t>(metrics.rows, 0x1FFFFFFF) * row_height));
      gui::set_cursor_pos(fan::vec2(gui::get_cursor_pos().x, first_visible_row * row_height + gui::get_style().WindowPadding.y));

      if (gui::begin_table("##hex_table", 3, gui::table_flags_sizing_fixed_fit | gui::table_flags_scroll_x)) {
        uint64_t end_row = std::min(first_visible_row + visible_row_count, metrics.rows);
        for (uint64_t r = first_visible_row; r < end_row; ++r) {
          if (config.group_size.y > 1 && r % config.group_size.y == 0 && r > first_visible_row) {
            gui::table_next_row();
            gui::dummy(fan::vec2(0, gui::get_text_line_height()));
          }

          gui::table_next_row();
          uint64_t row_start = r * config.cols;
          uint64_t row_end = std::min(row_start + config.cols, metrics.size);

          gui::table_next_column();
          gui::align_text_to_frame_padding();
          gui::text(config.col_text_addr, fan::to_hex(row_start, 8));

          gui::table_next_column();
          for (uint64_t idx = row_start; idx < row_end; ++idx) {
            f32_t pad = (idx + 1 < row_end) ? get_spacing(idx, row_end, true) : 0.f;
            render_cell(data, idx, metrics.cell_w, 0.f, dragging, true);
            if (idx + 1 < row_end) gui::same_line(0.f, pad);
          }

          gui::table_next_column();
          {
            gui::style_scope_t ss;
            ss.var(gui::style_var_item_spacing, fan::vec2(0.f, 0.f));
            ss.var(gui::style_var_frame_padding, fan::vec2(0.f, gui::get_frame_padding().y));
            gui::align_text_to_frame_padding();
            for (uint64_t idx = row_start; idx < row_end; ++idx) {
              f32_t pad = (idx + 1 < row_end) ? get_spacing(idx, row_end, false) : 0.f;
              render_cell(data, idx, metrics.ascii_w, 0.f, dragging, false);
              if (idx + 1 < row_end) gui::same_line(0.f, pad);
            }
          }
        }
        gui::end_table();
      }
    }

    gui::table_next_column();
    render_data_inspector(data, true);

    gui::end_table();
  }

  if (gui::is_window_hovered(0) && !prev_hex_hover_idx && !prev_ascii_hover_idx && fan::window::is_mouse_clicked(fan::mouse_left)) {
    sel_start = std::nullopt;
    sel_end = std::nullopt;
  }

  process_clipboard(data);

  gui::pop_font();
  gui::set_window_font_scale(1.0f);
}

  std::vector<uint8_t> hex_editor_t::get_selected_bytes(fan::io::data_provider_t& data) const {
    if (!has_selection()) return {};
    auto [lo, hi] = get_selection_bounds();
    uint64_t len = hi - lo + 1;

    std::vector<uint8_t> result;
    data.read_range(lo, len, result);
    return result;
  }

  std::optional<uint64_t> hex_editor_t::get_active_cell(fan::io::data_provider_t& data) const {
    if (!active_idx || active_idx.value() >= data.size()) return std::nullopt;
    return active_idx;
  }

  void hex_editor_t::process_clipboard(fan::io::data_provider_t& data) {
    bool ctrl = fan::window::is_key_down(fan::key_left_control);
    if (ctrl && fan::window::is_key_clicked(fan::key_c)) {
      fan::graphics::ctx().window->set_clipboard(fan::bytes2hex(get_selected_bytes(data)));
      return;
    }
    if (!ctrl || !fan::window::is_key_clicked(fan::key_v) || !has_selection()) return;

    auto [lo, hi] = get_selection_bounds();
    std::string clip = fan::graphics::ctx().window->get_clipboard();
    uint64_t next_idx = lo;

    if (active_panel == active_panel_t::ascii) {
      for (uint64_t i = 0; i < clip.size(); ++i) {
        uint64_t idx = lo + i;
        if (idx < metrics.size) {
          data.write(idx, (uint8_t)clip[i]);
          next_idx = idx + 1;
        }
      }
      pending_focus_ascii = next_idx;
    } else {
      auto bytes = fan::parse_hex_buffer(fan::trim(clip));
      for (uint64_t i = 0; i < bytes.size(); ++i) {
        uint64_t idx = lo + i;
        if (idx < metrics.size) {
          data.write(idx, bytes[i]);
          next_idx = idx + 1;
        }
      }
      pending_focus_hex = next_idx;
    }
    sel_start = next_idx;
    sel_end = next_idx;
  }
}
#endif