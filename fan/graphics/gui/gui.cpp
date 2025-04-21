#include <fan/graphics/gui/gui.h>

#if defined(fan_gui)

bool fan::graphics::gui::begin(const std::string& window_name, bool* p_open, window_flags_t window_flags) {
  return ImGui::Begin(window_name.c_str(), p_open, window_flags);
}

void fan::graphics::gui::end() {
  ImGui::End();
}

bool fan::graphics::gui::begin_child(const std::string& window_name, const fan::vec2& size, child_window_flags_t window_flags) {
  return ImGui::BeginChild(window_name.c_str(), size, window_flags);
}

void fan::graphics::gui::end_child(){
  ImGui::EndChild();
}

void fan::graphics::gui::same_line(f32_t offset_from_start_x, f32_t spacing_w) {
  ImGui::SameLine(offset_from_start_x, spacing_w);
}

void fan::graphics::gui::new_line() {
  ImGui::NewLine();
}

f32_t fan::graphics::gui::get_text_line_height_with_spacing() {
  return ImGui::GetTextLineHeightWithSpacing();
}

bool fan::graphics::gui::selectable(const std::string& label, bool selected, selectable_flag_t flags, const fan::vec2& size) {
  return ImGui::Selectable(label.c_str(), selected, flags, size);
}

bool fan::graphics::gui::selectable(const std::string& label, bool* p_selected, selectable_flag_t flags, const fan::vec2& size) {
  return ImGui::Selectable(label.c_str(), p_selected, flags, size);
}

fan::graphics::gui::window_t::window_t(const std::string& window_name, bool* p_open, window_flags_t window_flags)
  : is_open(fan::graphics::gui::begin(window_name.c_str(), p_open, window_flags)) {}

fan::graphics::gui::window_t::~window_t() {
  fan::graphics::gui::end();
}

fan::graphics::gui::window_t::operator bool() const {
  return is_open;
}

fan::graphics::gui::child_window_t::child_window_t(const std::string& window_name, const fan::vec2& size, child_window_flags_t window_flags)
  : is_open(ImGui::BeginChild(window_name.c_str(), size, window_flags)) {}
fan::graphics::gui::child_window_t::~child_window_t() {
  ImGui::EndChild();
}
fan::graphics::gui::child_window_t::operator bool() const {
  return is_open;
}

fan::graphics::gui::table_t::table_t(const std::string& str_id, int columns, table_flags_t flags, const fan::vec2& outer_size, f32_t inner_width) 
  : is_open(ImGui::BeginTable(str_id.c_str(), columns, flags, outer_size, inner_width)) {}
fan::graphics::gui::table_t::~table_t() {
  ImGui::EndTable();
}

fan::graphics::gui::table_t::operator bool() const {
  return is_open;
}

bool fan::graphics::gui::begin_table(const std::string& str_id, int columns, table_flags_t flags, const fan::vec2& outer_size, f32_t inner_width) {
  return ImGui::BeginTable(str_id.c_str(), columns, flags, outer_size, inner_width);
}

void fan::graphics::gui::end_table() {
  ImGui::EndTable();
}

void fan::graphics::gui::table_next_row(table_row_flags_t row_flags, f32_t min_row_height) {
  ImGui::TableNextRow(row_flags, min_row_height);
}

bool fan::graphics::gui::table_next_column() {
  return ImGui::TableNextColumn();
}

bool fan::graphics::gui::button(const std::string& label, const fan::vec2& size) {
  return ImGui::Button(label.c_str(), size);
}

void fan::graphics::gui::text(const std::string& text, const fan::color& color) {
  ImGui::PushStyleColor(ImGuiCol_Text, color);
  ImGui::Text("%s", text.c_str());
  ImGui::PopStyleColor();
}

void fan::graphics::gui::text_at(const std::string& text, const fan::vec2& position, const fan::color& color) {
  ImGui::SetCursorPos(position);
  ImGui::PushStyleColor(ImGuiCol_Text, color);
  ImGui::Text("%s", text.c_str());
  ImGui::PopStyleColor();
}
void fan::graphics::gui::text_bottom_right(const std::string& text, const fan::color& color, const fan::vec2& offset) {
  ImVec2 text_pos;
  ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
  ImVec2 window_pos = ImGui::GetWindowPos();
  ImVec2 window_size = ImGui::GetWindowSize();

  text_pos.x = window_pos.x + window_size.x - text_size.x - ImGui::GetStyle().WindowPadding.x;
  text_pos.y = window_pos.y + window_size.y - text_size.y - ImGui::GetStyle().WindowPadding.y;
  fan::graphics::gui::text_at(text, text_pos + offset, color);
}

bool fan::graphics::gui::drag_float(const std::string& label, f32_t* v, f32_t v_speed, f32_t v_min, f32_t v_max, const std::string& format, fan::graphics::gui::slider_flags_t flags) {
  return ImGui::DragFloat(label.c_str(), v, v_speed, v_min, v_max, format.c_str(), flags);
}

bool fan::graphics::gui::drag_float(const std::string& label, fan::vec2* v, f32_t v_speed, f32_t v_min, f32_t v_max, const std::string& format, fan::graphics::gui::slider_flags_t flags) {
  return ImGui::DragFloat2(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
}

bool fan::graphics::gui::drag_float(const std::string& label, fan::vec3* v, f32_t v_speed, f32_t v_min, f32_t v_max, const std::string& format, fan::graphics::gui::slider_flags_t flags) {
  return ImGui::DragFloat3(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
}

bool fan::graphics::gui::drag_float(const std::string& label, fan::vec4* v, f32_t v_speed, f32_t v_min, f32_t v_max, const std::string& format, fan::graphics::gui::slider_flags_t flags) {
  return ImGui::DragFloat4(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
}

bool fan::graphics::gui::drag_float(const std::string& label, fan::quat* q, f32_t v_speed, f32_t v_min, f32_t v_max, const std::string& format, fan::graphics::gui::slider_flags_t flags) {
  return ImGui::DragFloat4(label.c_str(), q->data(), v_speed, v_min, v_max, format.c_str(), flags);
}

bool fan::graphics::gui::drag_float(const std::string& label, fan::color* c, f32_t v_speed, f32_t v_min, f32_t v_max, const std::string& format, fan::graphics::gui::slider_flags_t flags) {
  return ImGui::DragFloat4(label.c_str(), c->data(), v_speed, v_min, v_max, format.c_str(), flags);
}

bool fan::graphics::gui::drag_int(const std::string& label, int* v, f32_t v_speed, int v_min, int v_max, const std::string& format, slider_flags_t flags) {
  return ImGui::DragInt(label.c_str(), v, v_speed, v_min, v_max, format.c_str(), flags);
}

bool fan::graphics::gui::drag_int(const std::string& label, fan::vec2i* v, f32_t v_speed, int v_min, int v_max, const std::string& format, slider_flags_t flags) {
  return ImGui::DragInt2(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
}

bool fan::graphics::gui::drag_int(const std::string& label, fan::vec3i* v, f32_t v_speed, int v_min, int v_max, const std::string& format, slider_flags_t flags) {
  return ImGui::DragInt3(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
}

bool fan::graphics::gui::drag_int(const std::string& label, fan::vec4i* v, f32_t v_speed, int v_min, int v_max, const std::string& format, slider_flags_t flags) {
  return ImGui::DragInt4(label.c_str(), v->data(), v_speed, v_min, v_max, format.c_str(), flags);
}

struct InputTextCallback_UserData {
  std::string*            Str;
  ImGuiInputTextCallback  ChainCallback;
  void*                   ChainCallbackUserData;
};

//imgui_stdlib.cpp:
static int InputTextCallback(ImGuiInputTextCallbackData* data) {
  InputTextCallback_UserData* user_data = (InputTextCallback_UserData*)data->UserData;
  if (data->EventFlag == ImGuiInputTextFlags_CallbackResize)
  {
      // Resize string callback
      // If for some reason we refuse the new length (BufTextLen) and/or capacity (BufSize) we need to set them back to what we want.
      std::string* str = user_data->Str;
      IM_ASSERT(data->Buf == str->c_str());
      str->resize(data->BufTextLen);
      data->Buf = (char*)str->c_str();
  }
  else if (user_data->ChainCallback)
  {
      // Forward to user callback, if any
      data->UserData = user_data->ChainCallbackUserData;
      return user_data->ChainCallback(data);
  }
  return 0;
}

bool fan::graphics::gui::input_text(const std::string& label, std::string* buf, input_text_flags_t flags, input_text_callback_t callback, void* user_data) {
  IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
  flags |= ImGuiInputTextFlags_CallbackResize;

  InputTextCallback_UserData cb_user_data;
  cb_user_data.Str = buf;
  cb_user_data.ChainCallback = callback;
  cb_user_data.ChainCallbackUserData = user_data;
  return ImGui::InputText(label.c_str(), (char*)buf->c_str(), buf->capacity() + 1, flags, InputTextCallback, &cb_user_data);
}

bool fan::graphics::gui::input_text_multiline(const std::string& label, std::string* buf, const ImVec2& size, input_text_flags_t flags, input_text_callback_t callback, void* user_data) {
  IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
  flags |= ImGuiInputTextFlags_CallbackResize;

  InputTextCallback_UserData cb_user_data;
  cb_user_data.Str = buf;
  cb_user_data.ChainCallback = callback;
  cb_user_data.ChainCallbackUserData = user_data;
  return ImGui::InputTextMultiline(label.c_str(), (char*)buf->c_str(), buf->capacity() + 1, size, flags, InputTextCallback, &cb_user_data);
}

bool fan::graphics::gui::input_float(const std::string& label, f32_t* v, f32_t step, f32_t step_fast, const char* format, input_text_flags_t flags) {
  return ImGui::InputFloat(label.c_str(), v, step, step_fast, format, flags);
}

bool fan::graphics::gui::input_float(const std::string& label, fan::vec2* v, const char* format, input_text_flags_t flags) {
  return ImGui::InputFloat2(label.c_str(), v->data(), format, flags);
}

bool fan::graphics::gui::input_float(const std::string& label, fan::vec3* v, const char* format, input_text_flags_t flags) {
  return ImGui::InputFloat3(label.c_str(), v->data(), format, flags);
}

bool fan::graphics::gui::input_float(const std::string& label, fan::vec4* v, const char* format, input_text_flags_t flags) {
  return ImGui::InputFloat4(label.c_str(), v->data(), format, flags);
}

bool fan::graphics::gui::input_int(const std::string& label, int* v, int step, int step_fast, input_text_flags_t flags) {
  return ImGui::InputInt(label.c_str(), v, step, step_fast, flags);
}

bool fan::graphics::gui::input_int(const std::string& label, fan::vec2i* v, input_text_flags_t flags) {
  return ImGui::InputInt2(label.c_str(), v->data(), flags);
}

bool fan::graphics::gui::input_int(const std::string& label, fan::vec3i* v, input_text_flags_t flags) {
  return ImGui::InputInt3(label.c_str(), v->data(), flags);
}

bool fan::graphics::gui::input_int(const std::string& label, fan::vec4i* v, input_text_flags_t flags) {
  return ImGui::InputInt4(label.c_str(), v->data(), flags);
}

bool fan::graphics::gui::color_edit3(const std::string& label, fan::color* color, color_edit_flags_t flags) {
  return ImGui::ColorEdit3(label.c_str(), color->data(), flags);
}

bool fan::graphics::gui::color_edit3(const std::string& label, fan::vec3* color, color_edit_flags_t flags) {
  return ImGui::ColorEdit3(label.c_str(), color->data(), flags);
}

bool fan::graphics::gui::color_edit4(const std::string& label, fan::color* color, color_edit_flags_t flags) {
  return ImGui::ColorEdit4(label.c_str(), color->data(), flags);
}

fan::vec2 fan::graphics::gui::get_window_size() {
  return ImGui::GetWindowSize();
}

void fan::graphics::gui::set_next_window_pos(const fan::vec2& position) {
  ImGui::SetNextWindowPos(position);
}

void fan::graphics::gui::set_next_window_size(const fan::vec2& size) {
  ImGui::SetNextWindowSize(size);
}

void fan::graphics::gui::push_style_color(col_t index, const fan::color& col) {
  ImGui::PushStyleColor(index, col);
}

void fan::graphics::gui::pop_style_color() {
  ImGui::PopStyleColor();
}

void fan::graphics::gui::push_style_var(style_var_t index, f32_t val) {
  ImGui::PushStyleVar(index, val);
}

void fan::graphics::gui::push_style_var(style_var_t index, const fan::vec2& val) {
  ImGui::PushStyleVar(index, val);
}

void fan::graphics::gui::pop_style_var() {
  ImGui::PopStyleVar();
}

template <typename T>
fan::graphics::gui::imgui_fs_var_t::imgui_fs_var_t(
  loco_t::shader_t shader_nr,
  const fan::string& var_name,
  T initial_,
  f32_t speed,
  f32_t min,
  f32_t max
) {
  //fan::vec_wrap_t < sizeof(T) / fan::conditional_value_t < std::is_class_v<T>, sizeof(T{} [0] ), sizeof(T) > , f32_t > initial = initial_;
  fan::vec_wrap_t<fan::conditional_value_t<std::is_arithmetic_v<T>, 1, sizeof(T) / sizeof(f32_t)>::value, f32_t> 
    initial;
  if constexpr (std::is_arithmetic_v<T>) {
    initial = (f32_t)initial_;
  }
  else {
    initial = initial_;
  }
  fan::opengl::context_t::shader_t shader = std::get<fan::opengl::context_t::shader_t>(gloco->shader_get(shader_nr));
  if (gloco->window.renderer == loco_t::renderer_t::vulkan) {
    fan::throw_error("");
  }
  auto found = gloco->shader_list[shader_nr].uniform_type_table.find(var_name);
  if (found == gloco->shader_list[shader_nr].uniform_type_table.end()) {
    //fan::print("failed to set uniform value");
    return;
    //fan::throw_error("failed to set uniform value");
  }
  ie = [str = found->second, shader_nr, var_name, speed, min, max, data = initial]() mutable {
    bool modify = false;
    switch(fan::get_hash(str)) {
      case fan::get_hash(std::string_view("float")): {
        modify = ImGui::DragFloat(fan::string(std::move(var_name)).c_str(), &data[0], (f32_t)speed, (f32_t)min, (f32_t)max);
        break;
      }
      case fan::get_hash(std::string_view("vec2")): {
        modify = ImGui::DragFloat2(fan::string(std::move(var_name)).c_str(), ((fan::vec2*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
        break;
      }
      case fan::get_hash(std::string_view("vec3")): {
        modify = ImGui::DragFloat3(fan::string(std::move(var_name)).c_str(), ((fan::vec3*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
        break;
      }
      case fan::get_hash(std::string_view("vec4")): {
        modify = ImGui::DragFloat4(fan::string(std::move(var_name)).c_str(), ((fan::vec4*)&data)->data(), (f32_t)speed, (f32_t)min, (f32_t)max);
        break;
      }
    }
    if (modify) {
      gloco->shader_set_value(shader_nr, var_name, data);
    }
  };
  gloco->shader_set_value(shader_nr, var_name, initial);
}


template fan::graphics::gui::imgui_fs_var_t::imgui_fs_var_t(
  loco_t::shader_t shader_nr,
  const fan::string& var_name,
  fan::vec2 initial_,
  f32_t speed,
  f32_t min,
  f32_t max
);
template fan::graphics::gui::imgui_fs_var_t::imgui_fs_var_t(
  loco_t::shader_t shader_nr,
  const fan::string& var_name,
  double initial_,
  f32_t speed,
  f32_t min,
  f32_t max
);

void fan::graphics::gui::set_viewport(fan::graphics::viewport_t viewport) {
  ImVec2 mainViewportPos = ImGui::GetMainViewport()->Pos;

  ImVec2 windowPos = ImGui::GetWindowPos();

  fan::vec2 windowPosRelativeToMainViewport;
  windowPosRelativeToMainViewport.x = windowPos.x - mainViewportPos.x;
  windowPosRelativeToMainViewport.y = windowPos.y - mainViewportPos.y;

  fan::vec2 window_size = gloco->window.get_size();
  fan::vec2 viewport_size = ImGui::GetContentRegionAvail();

  ImVec2 padding = ImGui::GetStyle().WindowPadding;
  viewport_size.x += padding.x * 2;
  viewport_size.y += padding.y * 2;

  fan::vec2 viewport_pos = fan::vec2(windowPosRelativeToMainViewport);
  gloco->viewport_set(
    viewport,
    viewport_pos,
    viewport_size,
    window_size
  );
}

fan::graphics::gui::imgui_element_nr_t::imgui_element_nr_t(const imgui_element_nr_t& nr) : imgui_element_nr_t() {
  if (nr.is_invalid()) {
    return;
  }
  init();
}

fan::graphics::gui::imgui_element_nr_t::imgui_element_nr_t(imgui_element_nr_t&& nr) {
  NRI = nr.NRI;
  nr.invalidate_soft();
}

fan::graphics::gui::imgui_element_nr_t::~imgui_element_nr_t() {
  invalidate();
}

fan::graphics::gui::imgui_element_nr_t& fan::graphics::gui::imgui_element_nr_t::operator=(const imgui_element_nr_t& id) {
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

fan::graphics::gui::imgui_element_nr_t& fan::graphics::gui::imgui_element_nr_t::operator=(imgui_element_nr_t&& id) {
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

void fan::graphics::gui::imgui_element_nr_t::init() {
  *(base_t*)this = fan::graphics::gui::m_imgui_draw_cb.NewNodeLast();
}

bool fan::graphics::gui::imgui_element_nr_t::is_invalid() const {
  return fan::graphics::gui::ns_imgui_draw::imgui_draw_cb_inric(*this);
}

void fan::graphics::gui::imgui_element_nr_t::invalidate_soft() {
  *(base_t*)this = fan::graphics::gui::m_imgui_draw_cb.gnric();
}

void fan::graphics::gui::imgui_element_nr_t::invalidate() {
  if (is_invalid()) {
    return;
  }
  fan::graphics::gui::m_imgui_draw_cb.unlrec(*this);
  *(base_t*)this = fan::graphics::gui::m_imgui_draw_cb.gnric();
}

const char* fan::graphics::gui::item_getter1(const std::vector<std::string>& items, int index) {
  if (index >= 0 && index < (int)items.size()) {
    return items[index].c_str();
  }
  return "N/A";
}

void fan::graphics::gui::process_loop() {
  auto it = fan::graphics::gui::m_imgui_draw_cb.GetNodeFirst();
  while (it != m_imgui_draw_cb.dst) {
    m_imgui_draw_cb.StartSafeNext(it);
    m_imgui_draw_cb[it]();
    it = m_imgui_draw_cb.EndSafeNext();
  }
}

bool fan::graphics::gui::audio_button(
  const std::string& label, 
   fan::audio::piece_t piece_hover, 
  fan::audio::piece_t piece_click, 
  const fan::vec2& size
) {
  ImGui::PushID(label.c_str());
  ImGuiStorage* storage = ImGui::GetStateStorage();
  ImGuiID id = ImGui::GetID("audio_button_prev_hovered");
  bool previously_hovered = storage->GetBool(id);
  
  bool pressed = ImGui::Button(label.c_str(), size);
  bool currently_hovered = ImGui::IsItemHovered();
  
  if (currently_hovered && !previously_hovered) {
    fan::audio::piece_t& piece = fan::audio::is_piece_valid(piece_hover) ? piece_hover : gloco->piece_hover;
    fan::audio::play(piece);
  }
  if (pressed) {
    fan::audio::piece_t& piece = fan::audio::is_piece_valid(piece_click) ? piece_click : gloco->piece_click;
    fan::audio::play(piece);
  }
  storage->SetBool(id, currently_hovered);

  ImGui::PopID();
  return pressed;
}

#endif