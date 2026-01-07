module;

#include <fan/utility.h>
#include <fan/event/types.h>

#include <string>
#include <functional>
#include <filesystem>
#include <coroutine>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <array>

module fan.graphics.gui;

import fan.types.vector;

import fan.graphics.gui.base;

#if defined(FAN_AUDIO)
import fan.audio;
#endif

#if defined(FAN_GUI)
namespace fan::graphics::gui {
  const char* item_getter1(const std::vector<std::string>& items, int index) {
    if (index >= 0 && index < (int)items.size()) {
      return items[index].c_str();
    }
    return "N/A";
  }

  void set_viewport(fan::graphics::viewport_t viewport) {
    fan::vec2 child_pos = get_window_pos();
    fan::vec2 child_size = get_window_size();
    fan::vec2 mainViewportPos = get_main_viewport()->Pos;

    fan::vec2 windowPosRelativeToMainViewport;
    windowPosRelativeToMainViewport.x = child_pos.x - mainViewportPos.x;
    windowPosRelativeToMainViewport.y = child_pos.y - mainViewportPos.y;

    fan::vec2 viewport_size = fan::vec2(child_size.x, child_size.y);
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
    fan::vec2 win = get_window_size();

    f32_t old_w = cam.coordinates.right  - cam.coordinates.left;
    f32_t old_h = cam.coordinates.top    - cam.coordinates.bottom;

    f32_t sx = win.x / old_w;
    f32_t sy = win.y / old_h;

    cam.coordinates.left   *= sx;
    cam.coordinates.right  *= sx;
    cam.coordinates.top    *= sy;
    cam.coordinates.bottom *= sy;

    fan::graphics::camera_set_ortho(
      render_view.camera,
      fan::vec2(cam.coordinates.left, cam.coordinates.right),
      fan::vec2(cam.coordinates.bottom, cam.coordinates.top)
    );
  }

  void image(fan::graphics::image_t img, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, const fan::color& tint_col, const fan::color& border_col) {
    image((texture_id_t)fan::graphics::image_get_handle(img), size, uv0, uv1, tint_col, border_col);
  }

  bool image_button(const std::string& str_id, fan::graphics::image_t img, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, int frame_padding, const fan::color& bg_col, const fan::color& tint_col) {
    return image_button(str_id.c_str(), (texture_id_t)fan::graphics::image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
  }

  bool image_text_button(fan::graphics::image_t img, const std::string& text, const fan::color& color, const fan::vec2& size, const fan::vec2& uv0, const fan::vec2& uv1, int frame_padding, const fan::color& bg_col, const fan::color& tint_col) {
    bool ret = image_button(text.c_str(), (texture_id_t)fan::graphics::image_get_handle(img), size, uv0, uv1, bg_col, tint_col);
    fan::vec2 text_size = calc_text_size(text.c_str());
    fan::vec2 min = get_item_rect_min();
    fan::vec2 pos = min + (get_item_rect_max() - min) / 2 - text_size / 2;
    get_window_draw_list()->AddText(pos, color.get_gui_color(), text.c_str());
    return ret;
  }

  bool toggle_image_button(const std::string& char_id, fan::graphics::image_t image, const fan::vec2& size, bool* toggle) {
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
      if (drag("count", &ri.count, 1)) {
      }
      if (slider("begin_angle", &ri.begin_angle, -fan::math::pi * 2, fan::math::pi * 2)) {
      }
      if (slider("end_angle", &ri.end_angle, -fan::math::pi * 2, fan::math::pi * 2)) {
      }
      if (slider("angle", &ri.angle, -fan::math::pi * 2, fan::math::pi * 2)) {
      }
      g_shapes->visit_shape_draw_data(shape.gint(), [&]<typename T>(T& properties) {
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


  content_browser_t::content_browser_t() {
    search_buffer.resize(32);
    current_directory = std::filesystem::path(asset_path);
    update_directory_cache();
  }

  content_browser_t::content_browser_t(bool no_init) {

  }

  content_browser_t::content_browser_t(const std::wstring& path) {
    init(path);
  }

  void content_browser_t::init(const std::wstring& path) {
    search_buffer.resize(32);
    current_directory = asset_path / std::filesystem::path(path);
    update_directory_cache();
  }

  void content_browser_t::clear_selection() {
    selection_state.selected_indices.clear();
    for (auto& file : directory_cache) {
      file.is_selected = false;
    }
    for (auto& file : search_state.found_files) {
      file.is_selected = false;
    }
  }

  bool content_browser_t::is_point_in_rect(const fan::vec2& point, const fan::vec2& rect_min, const fan::vec2& rect_max) {
    return point.x >= rect_min.x && point.x <= rect_max.x &&
      point.y >= rect_min.y && point.y <= rect_max.y;
  }

  void content_browser_t::handle_rectangular_selection() {
    io_t& io = get_io();
    selection_state.ctrl_held = io.KeyCtrl;

    if (fan::window::is_mouse_clicked()) {
      bool can_start_selection = !is_any_item_active() &&
        is_window_hovered() &&
        !is_any_item_hovered();

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

      fan::vec2 rect_min = fan::vec2(
        std::min(selection_state.selection_start.x, selection_state.selection_end.x),
        std::min(selection_state.selection_start.y, selection_state.selection_end.y)
      );
      fan::vec2 rect_max = fan::vec2(
        std::max(selection_state.selection_start.x, selection_state.selection_end.x),
        std::max(selection_state.selection_start.y, selection_state.selection_end.y)
      );

      if (showing_search_results) {
        update_search_sorted_cache();
      }
      else {
        update_sorted_cache();
      }
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
    while (!search_state.pending_directories.empty()) {
      search_state.pending_directories.pop();
    }

    std::fill(search_buffer.begin(), search_buffer.end(), '\0');

    clear_selection();

    invalidate_cache();

    for (auto& img : directory_cache) {
      if (fan::graphics::is_image_valid(img.preview_image)) {
        fan::graphics::image_unload(img.preview_image);
      }
    }
    directory_cache.clear();

    if (!directory_iterator.callback) {
      directory_iterator.sort_alphabetically = true;
      directory_iterator.callback = [this](const std::filesystem::directory_entry& entry) -> fan::event::task_t {
        file_info_t file_info;
        std::filesystem::path relative_path;
        try {
          // SLOW
          relative_path = std::filesystem::relative(entry, asset_path);
        }
        catch (const std::exception& e) {
          fan::print("exception came", e.what());
        }

        file_info.filename = relative_path.filename().generic_string();
        file_info.item_path = relative_path.wstring();
        file_info.is_directory = entry.is_directory();
        file_info.is_selected = false;
        //fan::print(get_file_extension(path.path().string()));
        if (!file_info.is_directory && fan::image::valid(entry.path().string())) {
          file_info.preview_image = fan::graphics::image_load(entry.path().generic_string());
        }
        directory_cache.push_back(file_info);
        invalidate_cache();
        co_return;
      };
    }

    fan::io::async_directory_iterate(
      &directory_iterator,
      current_directory.string()
    );
  }

  void content_browser_t::invalidate_cache() {
    search_state.cache_dirty = true;
    search_state.sorted_cache.clear();
  }

  int content_browser_t::get_pressed_key() {
    return gui::get_pressed_key();
  }

  void content_browser_t::handle_keyboard_navigation(const std::string& filename, int pressed_key) {
    if (pressed_key != -1 && !filename.empty()) {
      if (std::tolower(filename[0]) == std::tolower(pressed_key)) {
        set_scroll_here_y();
      }
    }
  }

  void content_browser_t::handle_right_click(const std::string& filename) {
    if (is_item_hovered() && fan::window::is_mouse_clicked(fan::mouse_right)) {
      item_right_clicked = true;
      item_right_clicked_name = filename;
      item_right_clicked_name.erase(
        std::remove_if(item_right_clicked_name.begin(), item_right_clicked_name.end(),
          [](unsigned char c) { return std::isspace(c); }),
        item_right_clicked_name.end()
      );
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
    search_state.found_files.clear();
    search_state.search_cache_dirty = true;

    while (!search_state.pending_directories.empty()) {
      search_state.pending_directories.pop();
    }

    if (query.empty()) {
      search_state.is_searching = false;
      search_state.found_files.clear();
      return;
    }

    search_state.is_searching = true;

    search_iterator.sort_alphabetically = true;
    search_iterator.callback = [this](const std::filesystem::directory_entry& entry) -> fan::event::task_t {
      try {
        std::string filename = entry.path().filename().string();

        std::string query_lower = search_state.query;
        std::string filename_lower = filename;
        static auto tl = [](unsigned char c) { return std::tolower(c); };
        std::transform(query_lower.begin(), query_lower.end(), query_lower.begin(), tl);
        std::transform(filename_lower.begin(), filename_lower.end(), filename_lower.begin(), tl);

        bool matches = filename_lower.find(query_lower) != std::string::npos;

        if (matches) {
          file_info_t file_info;
          std::filesystem::path relative_path;
          try {
            relative_path = std::filesystem::relative(entry.path(), asset_path);
          }
          catch (const std::exception&) {
            relative_path = entry.path().filename();
          }

          file_info.filename = filename;
          file_info.item_path = relative_path.wstring();
          file_info.is_directory = entry.is_directory();
          file_info.is_selected = false;

          if (fan::image::valid(entry.path().string())) {
            file_info.preview_image = fan::graphics::image_load(entry.path().generic_string());
          }

          search_state.found_files.push_back(file_info);
          search_state.search_cache_dirty = true;
        }

        if (entry.is_directory() && search_state.is_recursive) {
          search_state.pending_directories.push(entry.path().string());
        }
      }
      catch (...) {

      }

      co_return;
    };

    std::string search_root = current_directory.string();
    fan::io::async_directory_iterate(&search_iterator, search_root);
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
      [](const auto& a, const auto& b) {
      return a.first.is_directory > b.first.is_directory;
    });

    search_state.cache_dirty = false;
  }

  void content_browser_t::update_search_sorted_cache() {
    if (!search_state.search_cache_dirty) return;

    search_state.sorted_search_cache.clear();
    for (std::size_t i = 0; i < search_state.found_files.size(); ++i) {
      search_state.sorted_search_cache.push_back({search_state.found_files[i], i});
    }

    std::sort(search_state.sorted_search_cache.begin(), search_state.sorted_search_cache.end(),
      [](const auto& a, const auto& b) {
      return a.first.is_directory > b.first.is_directory;
    });

    search_state.search_cache_dirty = false;
  }

  void content_browser_t::render() {
    item_right_clicked = false;
    item_right_clicked_name.clear();

    if (search_state.is_searching && !search_iterator.operation_in_progress) {
      if (search_iterator.current_index >= search_iterator.entries.size()) {
        if (!search_state.pending_directories.empty()) {
          process_next_directory();
        }
        else {
          search_state.is_searching = false;
        }
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
          auto absolute_path = std::filesystem::canonical(std::filesystem::absolute(current_directory));
          if (absolute_path.has_parent_path()) {
            current_directory = std::filesystem::canonical(absolute_path.parent_path());
            update_directory_cache();
          }
        }
        same_line();
        image_button("##icon_arrow_right", icon_arrow_right, fan::vec2(32));
        same_line();
        pop_style_color(3);

        push_style_color(col_button, fan::color(0.f, 0.f, 0.f, 0.f));
        push_style_color(col_button_active, fan::color(0.f, 0.f, 0.f, 0.f));
        push_style_color(col_button_hovered, fan::color(0.3f, 0.3f, 0.3f, 0.3f));

        auto image_list = std::to_array({icon_files_list, icon_files_big_thumbnail});
        fan::vec2 bc = get_position_bottom_corner();
        bc.x -= get_window_pos().x;
        set_cursor_pos_x(bc.x / 2);

        fan::vec2 button_sizes = 32;
        set_next_item_width(get_content_region_avail().x - (button_sizes.x * 2 + style.ItemSpacing.x) * image_list.size());

        push_style_var(style_var_frame_rounding, 20.0f);
        push_style_var(style_var_frame_padding, fan::vec2(10.0f, 7.0f));
        f32_t y_pos = get_cursor_pos_y() + get_style().WindowPadding.y;
        set_cursor_pos_y(y_pos);

        static char old_search[256] = {0};
        bool search_changed = false;
        if (input_text("##content_browser_search", &search_buffer)) {
          search_changed = true;
        }

        if (search_changed || std::strcmp(old_search, search_buffer.data()) != 0) {
          strcpy(old_search, search_buffer.data());
          start_search(search_buffer.data(), true);
        }

        if (search_state.is_searching) {
          same_line();
          text("Searching... (%zu found)", search_state.found_files.size());
        }

        pop_style_var(2);
        toggle_image_button(image_list, button_sizes, (int*)&current_view_mode);
        pop_style_color(3);
        end_menu_bar();
      }

      handle_rectangular_selection();

      switch (current_view_mode) {
      case view_mode_large_thumbnails:
        render_large_thumbnails_view();
        break;
      case view_mode_list:
        render_list_view();
        break;
      default:
        break;
      }
    }

    pop_style_var();
    gui::end();
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

        std::string unique_id = "search_" + std::to_string(original_index) + "_" + file_info.filename;

        push_id(unique_id);

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
          "##" + unique_id,
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

          if (overlaps) {
            search_state.found_files[original_index].is_selected = true;
          }
          else if (!selection_state.ctrl_held) {
            search_state.found_files[original_index].is_selected = false;
          }
        }

        handle_right_click(file_info.filename);
        handle_item_interaction(file_info, original_index);

        pop_style_color(3);
        text_wrapped(file_info.filename);
        next_column();
        pop_id();
      }
    }
    else {
      update_sorted_cache();
      auto& sorted_files = search_state.sorted_cache;

      for (const auto& [file_info, original_index] : sorted_files) {
        handle_keyboard_navigation(file_info.filename, pressed_key);

        push_id(file_info.filename);

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

        bool item_clicked = image_button(
          "##" + file_info.filename,
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

          if (overlaps) {
            directory_cache[original_index].is_selected = true;
          }
          else if (!selection_state.ctrl_held) {
            directory_cache[original_index].is_selected = false;
          }
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
          while (calc_text_size(space.c_str()).x < image_size.x) {
            space += " ";
          }
          auto str = space + file_info.filename;

          std::string unique_id = "search_" + std::to_string(original_index) + "_" + str;

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

            if (overlaps) {
              search_state.found_files[original_index].is_selected = true;
            }
            else if (!selection_state.ctrl_held) {
              search_state.found_files[original_index].is_selected = false;
            }
          }

          handle_right_click(str);

          texture_id_t texture_id;
          if (fan::graphics::is_image_valid(file_info.preview_image)) {
            texture_id = (texture_id_t)fan::graphics::image_get_handle(file_info.preview_image);
          }
          else {
            texture_id = (texture_id_t)fan::graphics::image_get_handle(
              file_info.is_directory ? icon_directory : icon_file
            );
          }
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
          while (calc_text_size(space.c_str()).x < image_size.x) {
            space += " ";
          }
          auto str = space + file_info.filename;

          bool item_clicked = selectable(str, directory_cache[original_index].is_selected, selectable_flags_span_all_columns);

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

            if (overlaps) {
              directory_cache[original_index].is_selected = true;
            }
            else if (!selection_state.ctrl_held) {
              directory_cache[original_index].is_selected = false;
            }
          }

          handle_right_click(str);

          texture_id_t texture_id;
          if (fan::graphics::is_image_valid(file_info.preview_image)) {
            texture_id = (texture_id_t)fan::graphics::image_get_handle(file_info.preview_image);
          }
          else {
            texture_id = (texture_id_t)fan::graphics::image_get_handle(
              file_info.is_directory ? icon_directory : icon_file
            );
          }
          get_window_draw_list()->AddImage(texture_id, cursor_pos, cursor_pos + image_size);

          handle_item_interaction(file_info, original_index);
        }
      }

      end_table();
    }
  }

  void content_browser_t::handle_item_interaction(const file_info_t& file_info, size_t original_index) {
    if (file_info.is_directory == false) {
      if (begin_drag_drop_source()) {
        bool showing_search_results = !search_state.found_files.empty() && !search_buffer.empty();

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

        std::vector<std::wstring> selected_paths;

        if (showing_search_results) {
          for (const auto& f : search_state.found_files) {
            if (f.is_selected && !f.is_directory) {
              selected_paths.push_back(f.item_path);
            }
          }
        }
        else {
          for (const auto& f : directory_cache) {
            if (f.is_selected && !f.is_directory) {
              selected_paths.push_back(f.item_path);
            }
          }
        }

        if (selected_paths.empty()) {
          selected_paths.push_back(file_info.item_path);
        }

        std::wstring combined_paths;
        for (size_t i = 0; i < selected_paths.size(); ++i) {
          if (i > 0) combined_paths += L";";
          combined_paths += selected_paths[i];
        }
        set_drag_drop_payload("CONTENT_BROWSER_ITEMS", combined_paths.data(), (combined_paths.size() + 1) * sizeof(wchar_t));

        if (selected_paths.size() > 1) {
          text(selected_paths.size(), " files selected");
        }
        else {
          text(file_info.filename);
        }

        end_drag_drop_source();
      }
    }

    if (is_item_hovered() && is_mouse_double_clicked(0)) {
      if (file_info.is_directory) {
        current_directory = std::filesystem::path(asset_path) / file_info.item_path;
        update_directory_cache();
      }
    }
  }

  // [](const std::filesystem::path& path) {}
  void content_browser_t::receive_drag_drop_target(std::function<void(const std::filesystem::path& fs)> receive_func) {
    dummy(get_content_region_avail());

    if (begin_drag_drop_target()) {
      if (const payload_t* payload = accept_drag_drop_payload("CONTENT_BROWSER_ITEMS")) {
        const wchar_t* paths_data = (const wchar_t*)payload->Data;
        std::wstring combined_paths(paths_data);

        std::vector<std::filesystem::path> file_paths;
        std::wstring current_path;

        for (wchar_t c : combined_paths) {
          if (c == L';') {
            if (!current_path.empty()) {
              file_paths.push_back(std::filesystem::path(asset_path) / current_path);
              current_path.clear();
            }
          }
          else {
            current_path += c;
          }
        }

        if (!current_path.empty()) {
          file_paths.push_back(std::filesystem::path(asset_path) / current_path);
        }

        for (const auto& path : file_paths) {
          receive_func(std::filesystem::absolute(path));
        }
      }
      else if (const payload_t* payload = accept_drag_drop_payload("CONTENT_BROWSER_ITEMS")) {
        const wchar_t* path = (const wchar_t*)payload->Data;
        receive_func(std::filesystem::absolute(asset_path) / path);
      }
      end_drag_drop_target();
    }
  }

#if defined(FAN_2D)
  void fragment_shader_editor(uint16_t shape_type, std::string* fragment, bool* shader_compiled) {
    if (fragment->empty()) {
      *fragment = fan::graphics::shader_get_data(shape_type).sfragment;
    }
    if (begin("shader editor", 0, window_flags_no_saved_settings)) {
      if (!*shader_compiled) {
        gui::text("Failed to compile shader", fan::colors::red);
      }

      if (gui::input_text_multiline("##Shader Code", fragment, gui::get_content_region_avail(), gui::input_text_flags_allow_tab_input)) {
        *shader_compiled = fan::graphics::shader_update_fragment(shape_type, *fragment);
      }
      end();
    }
  }
#endif

  // called inside window begin end
  void animated_popup_window(const std::string& popup_id, const fan::vec2& popup_size, const fan::vec2& start_pos, const fan::vec2& target_pos, bool trigger_popup, std::function<void()> content_cb, const f32_t anim_duration, const f32_t hide_delay) {
    storage_t* storage = get_state_storage();
    id_t anim_time_id = get_id(popup_id + "_anim_time");
    id_t hide_timer_id = get_id(popup_id + "_hide_timer");
    id_t hovering_popup_id = get_id(popup_id + "_hovering");
    id_t popup_visible_id = get_id(popup_id + "_visible");

    f32_t delta_time = get_io().DeltaTime;

    f32_t popup_anim_time = storage->GetFloat(anim_time_id, 0.0f);
    f32_t hide_timer = storage->GetFloat(hide_timer_id, 0.0f);
    bool hovering_popup = storage->GetBool(hovering_popup_id, false);
    bool popup_visible = storage->GetBool(popup_visible_id, false);

    // Check if mouse is in parent window area
    //bool mouse_in_parent = (current_mouse_pos.x >= parent_min.x &&
    //  current_mouse_pos.x <= parent_max.x &&
    //  current_mouse_pos.y >= parent_min.y &&
    //  current_mouse_pos.y <= parent_max.y);

    if (trigger_popup || hovering_popup) {
      popup_visible = true;
      hide_timer = 0.0f;
    }
    else {
      hide_timer += delta_time;
    }

    if (popup_visible) {
      popup_anim_time = hide_timer < hide_delay ? std::min(popup_anim_time + delta_time, anim_duration)
        : std::max(popup_anim_time - delta_time, 0.0f);

      if (popup_anim_time == 0.0f) {
        popup_visible = false;
      }

      if (popup_anim_time > 0.0f) {
        f32_t t = popup_anim_time / anim_duration;
        t = t * t * (3.0f - 2.0f * t); // smoothstep

        // Simple interpolation between start and target positions
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
            hovered_flags_allow_when_blocked_by_active_item);

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

  bool sprite_animations_t::render_list_box(fan::graphics::animation_nr_t& shape_animation_id) {
    bool list_item_changed = false;

    gui::begin_child("animations_tool_bar", 0, 1);
    gui::set_cursor_pos_y(animation_names_padding);
    gui::indent(animation_names_padding);

    if (gui::button("+")) {
      fan::graphics::sprite_sheet_animation_t animation;
      animation.name = std::to_string((uint32_t)fan::graphics::all_animations_counter); // think this over
      shape_animation_id = fan::graphics::add_sprite_sheet_shape_animation(shape_animation_id, animation);
    }
    if (!shape_animation_id) {
      gui::unindent(animation_names_padding);
      gui::end_child();
      return false;
    }
    gui::push_item_width(gui::get_window_size().x * 0.8f);
    for (auto [i, animation_nr] : fan::enumerate(fan::graphics::get_sprite_sheet_shape_animation(shape_animation_id))) {
      auto& animation = fan::graphics::get_sprite_sheet_animation(animation_nr);
      if (animation.name == animation_list_name_to_edit) {
        std::snprintf(animation_list_name_edit_buffer.data(), animation_list_name_edit_buffer.size() + 1, "%s", animation.name.c_str());
        gui::push_id(i);
        if (set_focus) {
          gui::set_keyboard_focus_here();
          set_focus = false;
        }

        if (gui::input_text("##edit", &animation_list_name_edit_buffer, gui::input_text_flags_enter_returns_true)) {
          if (animation_list_name_edit_buffer != animation.name) {
            fan::graphics::rename_sprite_sheet_shape_animation(shape_animation_id, animation.name, animation_list_name_edit_buffer);
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
        if (gui::selectable(animation.name, current_animation_nr && current_animation_nr == animation_nr, gui::selectable_flags_allow_double_click, fan::vec2(gui::get_content_region_avail().x * 0.8f, 0))) {
          if (gui::is_mouse_double_clicked()) {
            animation_list_name_to_edit = animation.name;
            set_focus = true;
          }
          current_animation_shape_nr = shape_animation_id;
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

  bool sprite_animations_t::render_selectable_frames(fan::graphics::sprite_sheet_animation_t& current_animation) {
    bool changed = false;
    if (fan::window::is_mouse_released()) {
      previous_hold_selected.clear();
    }
    int grid_index = 0;
    bool first_button = true;

    for (int i = 0; i < current_animation.images.size(); ++i) {
      auto current_image = current_animation.images[i];
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
          auto& sf = current_animation.selected_frames;
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
            gui::text_outlined_at(std::to_string(std::distance(sf.begin(), it_found)), cursor_screen_pos);
            gui::text_at(std::to_string(std::distance(sf.begin(), it_found)), cursor_screen_pos);
          }
          gui::pop_id();

          if (!(y == vframes - 1 && x == hframes - 1 && i == current_animation.images.size() - 1)) {
            gui::same_line();
          }

          first_button = false;
          ++grid_index;
        }
      }
    }
    return changed;
  }

  bool sprite_animations_t::render(const std::string& drag_drop_id, fan::graphics::animation_nr_t& shape_animation_id) {
    gui::push_style_var(gui::style_var_item_spacing, fan::vec2(12.f, 12.f));
    gui::columns(2, "animation_columns", false);
    gui::set_column_width(0, gui::get_window_size().x * 0.2f);

    bool list_changed = render_list_box(shape_animation_id);

    gui::next_column();

    gui::begin_child("animation_window_right", 0, 1, gui::window_flags_horizontal_scrollbar);

    // just drop image from directory

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
    decltype(fan::graphics::all_animations)::iterator current_animation;
    if (!current_animation_nr) {
      goto g_end_frame;
    }
    current_animation = fan::graphics::all_animations.find(current_animation_nr);
    if (current_animation == fan::graphics::all_animations.end()) {
    g_end_frame:
      gui::columns(1);
      gui::end_child();
      gui::pop_style_var();
      return list_changed;
    }

    gui::same_line(0, 20.f);

    gui::slider_flags_t slider_flags = slider_flags_always_clamp | gui::slider_flags_no_speed_tweaks;
    list_changed |= gui::drag("fps", &current_animation->second.fps, 1, 0, 244, slider_flags);
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
          fan::print("Warning: drop target not valid (requires image file)");
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
                fan::vec2 uv_src = fan::vec2(
                  tc_size.x * x,
                  tc_size.y * y
                );
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
        if (auto it = fan::graphics::all_animations.find(current_animation_nr); it != fan::graphics::all_animations.end()) {
          auto& anim = it->second;
          fan::graphics::sprite_sheet_animation_t::image_t new_image;
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
    if (drag_drop_id.size()) {
      //fan::vec2 avail = gui::get_content_region_avail();
      fan::vec2 child_size = gui::get_window_size();
      dummy(child_size);
      gui::receive_drag_drop_target(drag_drop_id, [this](const std::string& file_paths) {
        for (const std::string& file_path : fan::split(file_paths, ";")) {
          if (fan::image::valid(file_path)) {
            if (auto it = fan::graphics::all_animations.find(current_animation_nr); it != fan::graphics::all_animations.end()) {
              auto& anim = it->second;
              //// unload previous image
              //if (fan::graphics::is_image_valid(anim.sprite_sheet)) {
              //  fan::graphics::image_unload(anim.sprite_sheet);
              //}
              fan::graphics::sprite_sheet_animation_t::image_t new_image;
              new_image.image = fan::graphics::image_load(file_path);
              anim.images.push_back(new_image);
            }
          }
          else {
            fan::print("Warning: drop target not valid (requires image file)");
          }
        }
      });
    }
    gui::set_cursor_pos(cursor_pos);

    //render_play_animation();

    list_changed |= render_selectable_frames(current_animation->second);

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

  void particle_editor_t::handle_file_operations() {
    if (open_file_dialog.is_finished()) {
      if (filename.size() != 0) {
        particle_shape = shape_from_json(filename);
        particle_image_sprite.set_image(particle_shape.get_image());
      }
      open_file_dialog.finished = false;
    }

    if (save_file_dialog.is_finished()) {
      if (filename.size() != 0) {
        fout(filename);
      }
      save_file_dialog.finished = false;
    }
  }

  void particle_editor_t::render_menu() {
    if (begin_main_menu_bar()) {
      if (begin_menu("File")) {
        if (menu_item("Open..", "Ctrl+O")) {
          open_file_dialog.load("json;fmm", &filename);
        }
        if (menu_item("Save as", "Ctrl+Shift+S")) {
          save_file_dialog.save("json;fmm", &filename);
        }
        end_menu();
      }
      end_main_menu_bar();
    }
  }

  void particle_editor_t::render_settings() {
    begin("particle settings");
    color_edit4("background color", &bg_color);
    gui::render_texture_property(particle_image_sprite, 0, "Particle texture");
    render_image_filter_property(particle_image_sprite, "Particle texture image filter");
    particle_shape.set_image(particle_image_sprite.get_image());
    shape_properties(particle_shape);
    end();

    if (fan::window::is_key_pressed(fan::key_s) && fan::window::is_key_down(fan::key_left_control)) {
      fout(filename);
    }
  }

  void particle_editor_t::render() {
    render_menu();
    handle_file_operations();
    render_settings();
  }

  void particle_editor_t::fout(const std::string& f) {
    filename = f;
    fan::json json_data = particle_shape;
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
        particle_image_sprite = {{.size = 0, .image = properties.image, .enable_culling=false}};
      }
    });
  }

#endif

  dialogue_box_t::render_type_t::~render_type_t() {}

  dialogue_box_t::text_delayed_t::~text_delayed_t() {
    dialogue_line_finished = true;
    character_advance_task = {};
  }

  void dialogue_box_t::text_delayed_t::render(dialogue_box_t* This, dialogue_box_t::drawable_nr_t nr, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {

    // initialize advance task but dont restart it after dialog finished
    if (dialogue_line_finished == false && !character_advance_task.owner) {
      character_advance_task = [This, nr]() -> fan::event::task_t {
        text_delayed_t* text_delayed = dynamic_cast<text_delayed_t*>(This->drawables[nr]);
        if (text_delayed == nullptr) {
          co_return;
        }

        // advance text rendering
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


      if (gui::button(text, size == 0 ? button_size : size)) {
        This->button_choice = nr;
        auto it = This->drawables.GetNodeFirst();
        while (it != This->drawables.dst) {
          This->drawables.StartSafeNext(it);
          if (dynamic_cast<button_t*>(This->drawables[it])) {
            delete This->drawables[it];
            This->drawables.unlrec(it);
          }
          it = This->drawables.EndSafeNext();
        }
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

  fan::event::task_value_resume_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::text_delayed(const std::string& character_name, const std::string& text) {
    return text_delayed(character_name, text, 20); // 20 characters per second
  }

  fan::event::task_value_resume_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::text_delayed(const std::string& character_name, const std::string& text, int characters_per_second) {
    text_delayed_t td;
    td.character_per_s = characters_per_second;
    td.text = text;
    td.render_pos = 0;
    td.dialogue_line_finished = false;

    auto it = drawables.NewNodeLast();
    drawables[it] = new text_delayed_t(std::move(td));

    co_return it;
  }

  fan::event::task_value_resume_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::text(const std::string& text) {
    text_t text_drawable;
    text_drawable.text = text;

    auto it = drawables.NewNodeLast();
    drawables[it] = new text_t(text_drawable);

    co_return it;
  }

  fan::event::task_value_resume_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::button(const std::string& text, const fan::vec2& position, const fan::vec2& size) {
    button_choice.sic();
    button_t button;
    button.position = position;
    button.size = size;
    button.text = text;

    auto it = drawables.NewNodeLast();
    drawables[it] = new button_t(button);
    co_return it;
  }

  // default width 80% of the window
  fan::event::task_value_resume_t<dialogue_box_t::drawable_nr_t> dialogue_box_t::separator(f32_t width) {
    auto it = drawables.NewNodeLast();
    drawables[it] = new separator_t;

    co_return it;
  }

  int dialogue_box_t::get_button_choice() {
    int btn_choice = -1;

    auto it = drawables.GetNodeFirst();
    while (it != drawables.dst) {
      drawables.StartSafeNext(it);
      if (dynamic_cast<button_t*>(drawables[it])) {
        if (button_choice == it) {
          break;
        }
      }
      it = drawables.EndSafeNext();
    }
    return btn_choice;
  }

  fan::event::task_t dialogue_box_t::wait_user_input() {
    wait_user = true;
    while (wait_user) {
      co_await fan::co_sleep(10);
    }
  }

  void dialogue_box_t::render(const std::string& window_name, font_t* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing, const std::function<void()>& inside_window_cb) {
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
    // render objects here

    auto it = drawables.GetNodeFirst();
    while (it != drawables.dst) {
      drawables.StartSafeNext(it);
      // co_await or task vector
      drawables[it]->render(this, it, window_size, wrap_width, line_spacing);
      it = drawables.EndSafeNext();
    }
    // end_child();
    set_window_font_scale(1.0f);

    bool dialogue_line_finished = fan::window::is_input_action_active("skip or continue dialog") && is_window_hovered(hovered_flags_child_windows | hovered_flags_allow_when_blocked_by_popup | hovered_flags_allow_when_blocked_by_active_item);

    if (dialogue_line_finished) {
      wait_user = false;
      clear();
    }

    end();
    pop_font();

  }

  void dialogue_box_t::render(const std::string& window_name, font_t* font, const fan::vec2& window_size, f32_t wrap_width, f32_t line_spacing) {
    render(window_name, font, window_size, wrap_width, line_spacing, [] {});
  }

  void dialogue_box_t::clear() {
    auto it = drawables.GetNodeFirst();
    while (it != drawables.dst) {
      drawables.StartSafeNext(it);
      delete drawables[it];
      drawables.unlrec(it);
      it = drawables.EndSafeNext();
    }
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

    // Add remaining text as last line
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
        text_unformatted(render_text.substr(0, render_pos - character_offset));
        break;
      }
      else {
        text_unformatted(render_text);
        if (render_text.back() != ' ') {
          character_offset += 1;
        }
        character_offset += render_text.size();
        pos.y += get_line_height_with_spacing() + line_spacing;
      }
    }
    if (empty_lines) {
      fan::graphics::gui::text(fan::colors::red, "warning empty lines:", empty_lines);
    }
  }

  void render_texture_property(
    fan::graphics::shape_t& shape, 
    int index, 
    const char* label,
    const std::wstring& asset_path,
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
  void render_image_filter_property(fan::graphics::shape_t& shape, const char* label) {
    using namespace fan::graphics;

    auto current_image = shape.get_image();
    int current_filter = fan::graphics::image_get_settings(current_image).min_filter;

    static const char* image_filters[] = { "nearest", "linear" };

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
}
#endif