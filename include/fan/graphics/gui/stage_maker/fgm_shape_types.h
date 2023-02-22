static std::size_t get_ending_bracket_offset(const fan::string& stage_name, const fan::string& str, std::size_t src) {
  std::size_t offset = src;
  std::size_t brackets_count = 0;

  do {
    std::size_t temp = str.find_first_of("{", offset);

    if (offset != fan::string::npos) {
      offset = temp + 1;
      brackets_count++;
    }
    else {
      fan::throw_error(fan::format("error processing {} . error at char:{}", stage_name, temp));
    }

    std::size_t s = str.find_first_of("{", offset);
    std::size_t d = str.find_first_of("}", offset);

    if (s < d) {
      continue;
    }
    offset = d + 1;
    brackets_count--;
  } while (brackets_count != 0);

  return offset;
}

template <std::size_t N>
void erase_cbs(pile_t* pile, const fan::string& file_name, const fan::string& stage_name, const fan::string& shape_name, auto* instance, const char* const(&cb_names)[N]) {
  fan::string str;
  fan::io::file::read(file_name, &str);

  const fan::string advance_str = fan::format("struct {0}_t", stage_name);
  auto advance_position = pile->stage_maker.stage_h_str.find(
    advance_str
  );

  if (advance_position == fan::string::npos) {
    fan::throw_error("corrupted stage.h:advance_position");
  }

  for (uint32_t j = 0; j < std::size(cb_names); ++j) {
    std::size_t src = str.find(
      fan::format("int {2}{0}_{1}_cb(const loco_t::{1}_data_t& mb)",
        instance->id, cb_names[j], shape_name)
    );

    if (src == fan::string::npos) {
      fan::throw_error("failed to find function:" + fan::format("int {3}{0}_{1}_cb(const loco_t::{1}_data_t& mb - from:{2})",
        instance->id, cb_names[j], stage_name, shape_name));
    }

    std::size_t dst = get_ending_bracket_offset(file_name, str, src);

    // - to remove endlines
    str.erase(src - 2, dst - src + 2);
  }

  fan::io::file::write(file_name, str, std::ios_base::binary);

  for (uint32_t j = 0; j < std::size(cb_names); ++j) {

    auto find_str = fan::format("{0}_{1}_cb_table_t {0}_{1}_cb_table[", shape_name, cb_names[j]);

    auto src = pile->stage_maker.stage_h_str.find(find_str, advance_position);
    if (src == fan::string::npos) {
      fan::throw_error("corrupted stage.h");
    }
    src += find_str.size();
    auto dst = pile->stage_maker.stage_h_str.find("]", src);
    if (dst == fan::string::npos) {
      fan::throw_error("corrupted stage.h");
    }
    fan::string tt = pile->stage_maker.stage_h_str.substr(src, dst - src);
    int val = std::stoi(tt);

    // prevent 0 sized array
    if (val != 1) {
      val -= 1;
    }
    pile->stage_maker.stage_h_str.replace(src, dst - src, std::to_string(val));

    find_str = fan::format("&{0}_t::{1}{2}_{3}_cb,", stage_name, shape_name, instance->id, cb_names[j]);

    src = pile->stage_maker.stage_h_str.find(find_str, advance_position);
    pile->stage_maker.stage_h_str.erase(src, find_str.size());
  }
  pile->stage_maker.write_stage();
}

struct line_t {

  #define fgm_shape_name line
  #define fgm_no_gui_properties
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    uint16_t shape;
  #include "fgm_shape_builder.h"

	void push_back(properties_t& p) {
		loco_t& loco = *get_loco();
    shape_builder_push_back
		loco.line.push_back(&instances[i]->cid, p);
	}

  fgm_make_clear_f(
    pile->loco.line.erase(&it->cid);
  );

  #include "fgm_shape_builder_close.h"
}line;

struct global_button_t {

  #define fgm_shape_name global_button
  #define fgm_no_gui_properties
  #define fgm_shape_loco_name button
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    uint16_t shape;
  #include "fgm_shape_builder.h"

	void push_back(properties_t& p) {
		p.position.z = 1;
		loco_t& loco = *get_loco();
		instances.resize(instances.size() + 1);
		uint32_t i = instances.size() - 1;
		instances[i] = new instance_t;
		instances[i]->shape = loco_t::shape_type_t::button;
		loco.button.push_back(&instances[i]->cid, p);
	}
  fgm_make_clear_f(
    pile->loco.button.erase(&it->cid);
  );

  #include "fgm_shape_builder_close.h"
}global_button;

fan::vec3 get_position(auto* shape, auto* instance) {
  return shape->get_position(instance);
}
void set_position(auto* shape, auto* instance, const fan::vec3& p) {
  shape->set_position(instance, p);
}

void move_shape(auto* shape, auto* instance, const fan::vec2& offset) {
  fan::vec3 p = get_position(shape, instance);
  p += fan::vec3(fan::vec2(
      matrices->coordinates.right - matrices->coordinates.left,
      matrices->coordinates.down - matrices->coordinates.up
    ) / get_loco()->get_window()->get_size() * offset, 0);
  set_position(shape, instance, p);
}

static bool does_id_exist(auto* shape, const fan::string& id) {
  for (const auto& it : shape->instances) {
    if (it->id == id) {
      return true;
    }
  }
  return false;
}

struct button_t {

  static constexpr const char* cb_names[] = { "mouse_button","mouse_move", "keyboard", "text" };

  struct properties_t : loco_t::button_t::properties_t {
    fan::string id;
  };

  uint8_t holding_special_key = 0;

  #define fgm_shape_name button
  #define fgm_shape_manual_properties
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    uint16_t shape; \
    loco_t::theme_t theme; \
    fan::string id;
  #include "fgm_shape_builder.h"

  void close_properties() {
    auto pile = get_pile();
    if (pile->stage_maker.fgm.properties_open) {
      pile->stage_maker.fgm.properties_open = false;
      pile->stage_maker.fgm.properties_nrs.clear();
      pile->stage_maker.fgm.text_box_menu.erase(pile->stage_maker.fgm.properties_nr);
    }
  }

	void open_properties(button_t::instance_t* instance) {
		auto pile = get_pile();

    close_properties();

    pile->stage_maker.fgm.properties_open = true;
    text_box_menu_t::open_properties_t menup;
		menup.matrices = &pile->stage_maker.fgm.matrices[pile_t::stage_maker_t::fgm_t::viewport_area::properties];
		menup.viewport = &pile->stage_maker.fgm.viewport[pile_t::stage_maker_t::fgm_t::viewport_area::properties];
		menup.theme = &pile->stage_maker.fgm.theme;
		menup.position = fan::vec2(0, -0.8);
		menup.gui_size = 0.08;
		auto nr = pile->stage_maker.fgm.text_box_menu.push_menu(menup);
		pile->stage_maker.fgm.properties_nr = nr;
    text_box_menu_t::properties_t p;
    auto position = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
		p.text = fan::format("{:.2f}, {:.2f}, {:.2f}", position.x, position.y, position.z);
		p.text_value = "add cbs";
		p.mouse_button_cb = [this, instance](const loco_t::mouse_button_data_t& mb) -> int {
			return 0;
		};
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[0]];
      auto text = pile->loco.text_box.get_text(&it.cid);
      fan::vec3 position;
      std::istringstream iss(fan::string(text).c_str());
      std::size_t i = 0;
      while (iss >> position[i++]) { iss.ignore(); }

      pile->loco.button.set_position(&instance->cid, position);
      pile->loco.button.set_depth(&instance->cid, position.z);
      
      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));

    auto size = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::size);
    p.text = fan::format("{:.2f}, {:.2f}", size.x, size.y);
    p.text_value = "add cbs";
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[1]];
      auto text = pile->loco.text_box.get_text(&it.cid);
      fan::vec2 size;
      std::istringstream iss(fan::string(text).c_str());
      std::size_t i = 0;
      while (iss >> size[i++]) { iss.ignore(); }

      pile->loco.button.set(&instance->cid, &loco_t::button_t::vi_t::size, size);

      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));

    const auto& text = pile->loco.button.get_text(&instance->cid);
    p.text = text;
    p.text_value = "text";
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[2]];
      auto text = pile->loco.text_box.get_text(&it.cid);
      pile->loco.button.set_text(&instance->cid, text);

      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));
	}
	void push_back(properties_t& p) {
		p.position.z = 1;
		pile_t* pile = get_pile();
    shape_builder_push_back

    if (p.id.empty()) {
      static uint32_t id = 0;
      while (does_id_exist(this, std::to_string(id))) { ++id; }
      instances[i]->id = std::to_string(id);
    }
    else {
      instances[i]->id = p.id;
    }

		instances[i]->theme = *(loco_t::theme_t*)pile->loco.get_context()->theme_list[p.theme].theme_id;
		p.mouse_button_cb = [this, instance = instances[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
			pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
      switch (ii_d.button) {
        case fan::mouse_scroll_up: {
          if (pile->stage_maker.fgm.action_flag & action::move) {
            fan::vec3 ps = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
            ps.z += 0.5;
            pile->loco.button.set_position(&instance->cid, ps);
            pile->loco.button.set_depth(&instance->cid, ps.z);
            pile->stage_maker.fgm.button.open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_scroll_down: {
          if (pile->stage_maker.fgm.action_flag & action::move) {
            fan::vec3 ps = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
            ps.z -= 0.5;
            ps.z = fan::clamp((f32_t)ps.z, (f32_t)0.f, (f32_t)ps.z);
            pile->loco.button.set_position(&instance->cid, ps);
            pile->loco.button.set_depth(&instance->cid, ps.z);
            pile->stage_maker.fgm.button.open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_left:{
          break;
        }
        default: {
          return 0;
        }
      }
			if (ii_d.button_state == fan::mouse_state::release) {
				pile->stage_maker.fgm.button.release();
				// TODO FIX, erases in near bottom
				if (!pile->stage_maker.fgm.viewport[viewport_area::editor].inside(pile->loco.get_mouse_position()) && !holding_special_key) {
					pile->stage_maker.fgm.button.erase(instance);
				}
				return 0;
			}
			if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
				return 0;
			}
			pile->stage_maker.fgm.action_flag |= action::move;
			auto viewport = pile->loco.button.get_viewport(&instance->cid);
			pile->stage_maker.fgm.click_position = ii_d.position;
			pile->stage_maker.fgm.move_offset = fan::vec2(pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position)) - pile->stage_maker.fgm.click_position;
			pile->stage_maker.fgm.resize_offset = pile->stage_maker.fgm.click_position;
			fan::vec3 rp = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
			fan::vec3 rs = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::size);
			pile->stage_maker.fgm.resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
			pile->stage_maker.fgm.button.open_properties(instance);
			return 0;
		};
		p.mouse_move_cb = [this, instance = instances[i]](const loco_t::mouse_move_data_t& ii_d) -> int {

      	if (ii_d.flag->ignore_move_focus_check == false) {
    return 0;
  }
  pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
  if (!(pile->stage_maker.fgm.action_flag & action::move)) {
    return 0;
  }

  if (holding_special_key) {
    fan::vec3 ps = get_position(instance);
    fan::vec2 rs = get_size(instance);

    static constexpr f32_t minimum_rectangle_size = 0.03;
    static constexpr fan::vec2i multiplier[] = { {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };

    rs += (ii_d.position - pile->stage_maker.fgm.resize_offset) * multiplier[pile->stage_maker.fgm.resize_side] / 2;

    if (rs.x == minimum_rectangle_size && rs.y == minimum_rectangle_size) {
      pile->stage_maker.fgm.resize_offset = ii_d.position;
    }

    bool ret = 0;
    if (rs.y < minimum_rectangle_size) {
      rs.y = minimum_rectangle_size;
      if (!(rs.x < minimum_rectangle_size)) {
        ps.x += (ii_d.position.x - pile->stage_maker.fgm.resize_offset.x) / 2;
        pile->stage_maker.fgm.resize_offset.x = ii_d.position.x;
      }
      ret = 1;
    }
    if (rs.x < minimum_rectangle_size) {
      rs.x = minimum_rectangle_size;
      if (!(rs.y < minimum_rectangle_size)) {
        ps.y += (ii_d.position.y - pile->stage_maker.fgm.resize_offset.y) / 2;
        pile->stage_maker.fgm.resize_offset.y = ii_d.position.y;
      }
      ret = 1;
    }

    if (rs != minimum_rectangle_size) {
      ps += (ii_d.position - pile->stage_maker.fgm.resize_offset) / 2;
    }
    if (rs.x == minimum_rectangle_size && rs.y == minimum_rectangle_size) {
      ps = get_position(instance);
    }

    set_size(instance, rs);
    set_position(instance, ps);

    if (ret) {
      return 0;
    }

    pile->stage_maker.fgm.resize_offset = ii_d.position;
    pile->stage_maker.fgm.move_offset = ps - fan::vec3(ii_d.position, 0);
    pile->stage_maker.fgm.fgm_shape_name.open_properties(instance);
    return 0;
  }

  fan::vec3 ps = get_position(instance);
  fan::vec3 p;
  p.x = ii_d.position.x + pile->stage_maker.fgm.move_offset.x;
  p.y = ii_d.position.y + pile->stage_maker.fgm.move_offset.y;
  p.z = ps.z;
  set_position(instance, p);

  pile->stage_maker.fgm.fgm_shape_name.open_properties(instance);

			return 0;
		};
		p.keyboard_cb = [this, instance = instances[i]](const loco_t::keyboard_data_t& kd) -> int {
			pile_t* pile = OFFSETLESS(OFFSETLESS(kd.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);

			switch (kd.key) {
				case fan::key_delete: {
          if (kd.keyboard_state != fan::keyboard_state::press) {
            return 0;
          }
					switch (kd.keyboard_state) {
						case fan::keyboard_state::press: {
							pile->stage_maker.fgm.button.erase(instance);
							pile->stage_maker.fgm.invalidate_focus();
							break;
						}
					}
					break;
				}
				case fan::key_c: {
					holding_special_key = kd.keyboard_state == fan::keyboard_state::release ? 0 : 1;
					break;
				}
        case fan::key_left:
        case fan::key_right:
        case fan::key_up:
        case fan::key_down:
        {
          if (kd.keyboard_state == fan::keyboard_state::release) {
            return 0;
          }
          if (kd.key == fan::key_left) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(-1, 0));
          if (kd.key == fan::key_right) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(1, 0));
          if (kd.key == fan::key_up) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(0, -1));
          if (kd.key == fan::key_down) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(0, 1));
          open_properties(instance);
          break;
        }
			}
			return 0;
		};
		pile->loco.button.push_back(&instances[i]->cid, p);
		pile->loco.button.set_theme(&instances[i]->cid, loco_t::button_t::released);
		auto builder_cid = &instances[i]->cid;
		auto ri = pile->loco.button.get_ri(builder_cid);
		pile->loco.vfi.set_focus_mouse(ri.vfi_id);
	}
	void erase(instance_t* instance) {
		pile_t* pile = OFFSETLESS(get_loco(), pile_t, loco_var_name);

    close_properties();

		pile->loco.button.erase(&instance->cid);

    auto stage_name = pile->stage_maker.get_selected_name(
      pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id
    );
    auto file_name = pile->stage_maker.get_file_fullpath(stage_name);

    pile->stage_maker.fgm.erase_cbs(pile, file_name, stage_name, "button", instance, cb_names);

    for (uint32_t i = 0; i < instances.size(); i++) {
      if (&instances[i]->cid == &instance->cid) {
        instances.erase(instances.begin() + i);
        break;
      }
    }
		release();
	}

  fgm_make_clear_f(
    pile->loco.button.erase(&it->cid);
  );

  fan::vec2 get_size(instance_t* instance) {
    return get_loco()->button.get(&instance->cid, &loco_t::button_t::vi_t::size);
  }
  void set_size(instance_t* instance, const fan::vec2& size) {
    get_loco()->button.set_size(&instance->cid, size);
  }

  fan::vec3 get_position(instance_t* instance) {
    return get_loco()->button.get(&instance->cid, &loco_t::button_t::vi_t::position);
  }
  void set_position(instance_t* instance, const fan::vec3& position) {
    get_loco()->button.set_position(&instance->cid, position);
  }

  fan::string to_string() const {
    auto* loco = get_loco();
    fan::string f;
    uint32_t instances_count = instances.size();
    fan::write_to_string(f, loco_t::shape_type_t::button);
    fan::write_to_string(f, instances_count);
    for (auto it : instances) {
      stage_maker_shape_format::shape_button_t data;
      data.position = loco->button.get(&it->cid, &loco_t::button_t::vi_t::position);
      data.size = loco->button.get(&it->cid, &loco_t::button_t::vi_t::size);
      data.font_size = loco->text.get_instance(
        &loco->button.get_ri(&it->cid).text_id
      ).font_size;
      data.text = loco->button.get_text(&it->cid);
      data.id = it->id;
      f += shape_to_string(data);
    }
    return f;
  }
  uint64_t from_string(const fan::string& f) {
    auto* pile = get_pile();
    uint64_t off = 0;
    loco_t::shape_type_t::_t shape_type = fan::read_data<loco_t::shape_type_t::_t>(f, off);
    if (shape_type != loco_t::shape_type_t::fgm_shape_loco_name) {
      return 0;
    }
    uint32_t instance_count = fan::read_data<uint32_t>(f, off);
    for (uint32_t i = 0; i < instance_count; ++i) {
      stage_maker_shape_format::shape_button_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
      });
      button_t::properties_t bp;
      bp.position = data.position;
      bp.size = data.size;
      bp.font_size = data.font_size;
      bp.text = data.text;
      bp.theme = &pile->stage_maker.fgm.theme;
      bp.matrices = &pile->stage_maker.fgm.matrices[viewport_area::editor];
      bp.viewport = &pile->stage_maker.fgm.viewport[viewport_area::editor];
      bp.id = data.id;
      push_back(bp);
    }
    return off;
  }

  #include "fgm_shape_builder_close.h"
};

struct sprite_t {
  struct properties_t : loco_t::sprite_t::properties_t {
    fan::string id;
    fan::string texturepack_name;
  };
  uint8_t holding_special_key = 0;

  #define fgm_shape_name sprite
  #define fgm_shape_manual_properties
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    loco_t::vfi_t::shape_id_t vfi_id; \
    uint16_t shape; \
    fan::string texturepack_name; \
    fan::string id;
  #include "fgm_shape_builder.h"

  void close_properties() {
    auto pile = get_pile();
    if (pile->stage_maker.fgm.properties_open) {
      pile->stage_maker.fgm.properties_open = false;
      pile->stage_maker.fgm.properties_nrs.clear();
      pile->stage_maker.fgm.text_box_menu.erase(pile->stage_maker.fgm.properties_nr);
    }
  }

	void open_properties(instance_t* instance) {
		auto pile = get_pile();

    close_properties();
    pile->stage_maker.fgm.properties_open = true;

		text_box_menu_t::open_properties_t menup;
		menup.matrices = &pile->stage_maker.fgm.matrices[pile_t::stage_maker_t::fgm_t::viewport_area::properties];
		menup.viewport = &pile->stage_maker.fgm.viewport[pile_t::stage_maker_t::fgm_t::viewport_area::properties];
		menup.theme = &pile->stage_maker.fgm.theme;
		menup.position = fan::vec2(0, -0.8);
		menup.gui_size = 0.08;
		auto nr = pile->stage_maker.fgm.text_box_menu.push_menu(menup);
		pile->stage_maker.fgm.properties_nr = nr;
    text_box_menu_t::properties_t p;
    auto position = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
    p.text = fan::format("{:.2f}, {:.2f}, {:.2f}", position.x, position.y, position.z);
    p.text_value = "add cbs";
		p.mouse_button_cb = [this, instance](const loco_t::mouse_button_data_t& mb) -> int {
			use_key_lambda(fan::mouse_left, fan::mouse_state::release);

			auto pile = get_pile();

			// open cb here

			return 0;
		};
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[0]];
      auto text = pile->loco.text_box.get_text(&it.cid);
      
      fan::vec3 position;
      std::istringstream iss(fan::string(text).c_str());
      std::size_t i = 0;
      while (iss >> position[i++]) { iss.ignore(); }

      get_loco()->vfi.shape_list[instance->vfi_id].shape_data.shape.rectangle.position = position;
      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
      pile->loco.sprite.sb_set_depth(&instance->cid, position.z);

      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));

    auto size = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
    p.text = fan::format("{:.2f}, {:.2f}", size.x, size.y);
    p.text_value = "";
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[1]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec2 size;
      std::istringstream iss(fan::string(text).c_str());
      std::size_t i = 0;
      while (iss >> size[i++]) { iss.ignore(); }

      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);
      get_loco()->vfi.shape_list[instance->vfi_id].shape_data.shape.rectangle.size = size;

      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));

    p.text = instance->texturepack_name;
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[2]];
      auto text = pile->loco.text_box.get_text(&it.cid);
      
      loco_t::texturepack_t::ti_t ti;
      if (pile->stage_maker.fgm.texturepack.qti(text, &ti)) {
        fan::print_no_space("failed to load texture:", fan::string(text).c_str());
        return 0;
      }
      auto& data = pile->stage_maker.fgm.texturepack.get_pixel_data(ti.pack_id);
      pile->loco.sprite.set_image(&instance->cid, &data.image);
      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::tc_position, ti.position / data.image.size);
      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::tc_size, ti.size / data.image.size);
      instance->texturepack_name = text;
      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));

    auto parallax_factor = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::parallax_factor);
    p.text = fan::format("{:.2f}", parallax_factor);
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[3]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      f32_t parallax;
      std::istringstream iss(fan::string(text).c_str());
      while (iss >> parallax) { iss.ignore(); }

      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::parallax_factor, parallax);
      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));
	}
	void push_back(properties_t& p) {
		p.position.z = 1;
		pile_t* pile = get_pile();
    shape_builder_push_back
    instances[i]->texturepack_name = p.texturepack_name;

    if (p.id.empty()) {
      static uint32_t id = 0;
      while (does_id_exist(this, std::to_string(id))) { ++id; }
      instances[i]->id = std::to_string(id);
    }
    else {
      instances[i]->id = p.id;
    }

		loco_t::vfi_t::properties_t vfip;
		vfip.mouse_button_cb = [pile, this, instance = instances[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
      switch (ii_d.button) {
        case fan::mouse_scroll_up: {
          if (pile->stage_maker.fgm.action_flag & action::move) {
            fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
            ps.z += 0.5;
            pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
            pile->loco.sprite.sb_set_depth(&instance->cid, ps.z);
            get_loco()->vfi.shape_list[instance->vfi_id].shape_data.depth = ps.z;
            open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_scroll_down: {
          if (pile->stage_maker.fgm.action_flag & action::move) {
            fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
            ps.z -= 0.5;
            ps.z = fan::clamp((f32_t)ps.z, (f32_t)0.f, (f32_t)ps.z);
            pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
            pile->loco.sprite.sb_set_depth(&instance->cid, ps.z);
            get_loco()->vfi.shape_list[instance->vfi_id].shape_data.depth = ps.z;
            open_properties(instance);
          }
          return 0;
        }
        case fan::mouse_left: {
          break;
        }
        default: {
          return 0;
        }
      }
			if (ii_d.button_state == fan::mouse_state::press) {
				ii_d.flag->ignore_move_focus_check = true;
				ii_d.vfi->set_focus_keyboard(ii_d.vfi->get_focus_mouse());
			}
			if (ii_d.button_state == fan::mouse_state::release) {
				ii_d.flag->ignore_move_focus_check = false;
			}
			pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);

			if (ii_d.button_state == fan::mouse_state::release) {
				release();
				// TODO FIX, erases in near bottom
				if (!pile->stage_maker.fgm.viewport[viewport_area::editor].inside(pile->loco.get_mouse_position()) && !holding_special_key) {
					erase(&instance->cid);
				}
				return 0;
			}
			if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
				return 0;
			}
			pile->stage_maker.fgm.action_flag |= action::move;
			auto viewport = pile->loco.sprite.get_viewport(&instance->cid);
			pile->stage_maker.fgm.click_position = ii_d.position;
			pile->stage_maker.fgm.move_offset = fan::vec2(pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position)) - pile->stage_maker.fgm.click_position;
			pile->stage_maker.fgm.resize_offset = pile->stage_maker.fgm.click_position;
			fan::vec3 rp = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
			fan::vec3 rs = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
			pile->stage_maker.fgm.resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
			open_properties(instance);
			return 0;
		};
		vfip.mouse_move_cb = [this, instance = instances[i]](const loco_t::mouse_move_data_t& ii_d) -> int {
      create_shape_move_resize
			return 0;
		};
		vfip.keyboard_cb = [this, i](const loco_t::keyboard_data_t& kd) -> int {
			pile_t* pile = OFFSETLESS(OFFSETLESS(kd.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
      auto* instance = instances[i];
			switch (kd.key) {
				case fan::key_delete: {
          if (kd.keyboard_state != fan::keyboard_state::press) {
            return 0;
          }
				switch (kd.keyboard_state) {
					case fan::keyboard_state::press: {
						erase(&instances[i]->cid);
						pile->stage_maker.fgm.invalidate_focus();
						break;
					}
				}
				break;
			}
			case fan::key_c: {
				holding_special_key = kd.keyboard_state == fan::keyboard_state::release ? 0 : 1;
				break;
			}
      case fan::key_left:
      case fan::key_right:
      case fan::key_up:
      case fan::key_down:
      {
        if (kd.keyboard_state == fan::keyboard_state::release) {
          return 0;
        }
        if (kd.key == fan::key_left) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(-1, 0));
        if (kd.key == fan::key_right) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(1, 0));
        if (kd.key == fan::key_up) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(0, -1));
        if (kd.key == fan::key_down) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(0, 1));
        open_properties(instance);
        break;
      }
			}
			return 0;
		};
		vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
		vfip.shape.rectangle.position = p.position;
		vfip.shape.rectangle.size = p.size;
		vfip.shape.rectangle.matrices = p.matrices;
		vfip.shape.rectangle.viewport = p.viewport;
		pile->loco.push_back_input_hitbox(&instances[i]->vfi_id, vfip);
		pile->loco.sprite.push_back(&instances[i]->cid, p);
	}
	void erase(fan::graphics::cid_t* cid) {
		loco_t& loco = *get_loco();
    close_properties();
		loco.sprite.erase(cid);
		for (uint32_t i = 0; i < instances.size(); i++) {
			if (&instances[i]->cid == cid) {
				instances.erase(instances.begin() + i);
				break;
			}
		}
		release();
	}

  fgm_make_clear_f(
    pile->loco.sprite.erase(&it->cid);
    pile->loco.vfi.erase(&it->vfi_id);
  );

  fan::vec3 get_position(instance_t* instance) {
    auto pile = get_pile();
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
  }
	void set_position(instance_t* instance, const fan::vec3& position) {
		auto pile = get_pile();
		pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
		pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::position, position);
	}

  fan::vec2 get_size(instance_t* instance) {
    auto pile = get_pile();
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
  }
	void set_size(instance_t* instance, const fan::vec2& size) {
		auto pile = get_pile();
		pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);
		pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::size, size);
	}

  fan::string to_string() const {
    auto* loco = get_loco();
    fan::string f;
    uint32_t instances_count = instances.size();
    fan::write_to_string(f, loco_t::shape_type_t::sprite);
    fan::write_to_string(f, instances_count);
    for (auto it : instances) {
      stage_maker_shape_format::shape_sprite_t data;
      data.position = loco->sprite.get(&it->cid, &loco_t::sprite_t::vi_t::position);
      data.size = loco->sprite.get(&it->cid, &loco_t::sprite_t::vi_t::size);
      data.parallax_factor = loco->sprite.get(&it->cid, &loco_t::sprite_t::vi_t::parallax_factor);
      data.texturepack_name = it->texturepack_name;
      data.id = it->id;
      f += shape_to_string(data);
    }
    return f;
  }

  uint64_t from_string(const fan::string& f) {
    auto* pile = get_pile();
    uint64_t off = 0;
    loco_t::shape_type_t::_t shape_type = fan::read_data<loco_t::shape_type_t::_t>(f, off);
    if (shape_type != loco_t::shape_type_t::fgm_shape_loco_name) {
      return 0;
    }
    uint32_t instance_count = fan::read_data<uint32_t>(f, off);
    for (uint32_t i = 0; i < instance_count; ++i) {
      stage_maker_shape_format::shape_sprite_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
        });
      sprite_t::properties_t sp;
      sp.position = data.position;
      sp.size = data.size;
      sp.parallax_factor = data.parallax_factor;
      loco_t::texturepack_t::ti_t ti;
      if (pile->stage_maker.fgm.texturepack.qti(data.texturepack_name, &ti)) {
        sp.image = &get_loco()->default_texture;
      }
      else {
        auto& pd = pile->stage_maker.fgm.texturepack.get_pixel_data(ti.pack_id);
        sp.image = &pd.image;
        sp.tc_position = ti.position / pd.image.size;
        sp.tc_size = ti.size / pd.image.size;
      }
      sp.matrices = &pile->stage_maker.fgm.matrices[viewport_area::editor];
      sp.viewport = &pile->stage_maker.fgm.viewport[viewport_area::editor];
      sp.texturepack_name = data.texturepack_name;
      sp.id = data.id;
      push_back(sp);
    }
    return off;
  }

  #include "fgm_shape_builder_close.h"
};

struct text_t {
  uint8_t holding_special_key = 0;

  struct properties_t : loco_t::text_t::properties_t {
    fan::string id;
  };

  #define fgm_shape_name text
  #define fgm_shape_manual_properties
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    loco_t::vfi_t::shape_id_t vfi_id; \
    uint16_t shape; \
    fan::string texturepack_name; \
    fan::string id;
  #include "fgm_shape_builder.h"

  void close_properties() {
    auto pile = get_pile();
    if (pile->stage_maker.fgm.properties_open) {
      pile->stage_maker.fgm.properties_open = false;
      pile->stage_maker.fgm.properties_nrs.clear();
      pile->stage_maker.fgm.text_box_menu.erase(pile->stage_maker.fgm.properties_nr);
    }
  }

  void open_properties(text_t::instance_t* instance) {
    auto pile = get_pile();

    close_properties();
    pile->stage_maker.fgm.properties_open = true;

    text_box_menu_t::open_properties_t menup;
    menup.matrices = &pile->stage_maker.fgm.matrices[pile_t::stage_maker_t::fgm_t::viewport_area::properties];
    menup.viewport = &pile->stage_maker.fgm.viewport[pile_t::stage_maker_t::fgm_t::viewport_area::properties];
    menup.theme = &pile->stage_maker.fgm.theme;
    menup.position = fan::vec2(0, -0.8);
    menup.gui_size = 0.08;
    auto nr = pile->stage_maker.fgm.text_box_menu.push_menu(menup);
    pile->stage_maker.fgm.properties_nr = nr;
    text_box_menu_t::properties_t p;
    auto position = pile->loco.text.get_instance(&instance->cid).position;
    p.text = fan::format("{:.2f}, {:.2f}, {:.2f}", position.x, position.y, position.z);
    p.text_value = "add cbs";
    p.mouse_button_cb = [this, instance](const loco_t::mouse_button_data_t& mb) -> int {
      use_key_lambda(fan::mouse_left, fan::mouse_state::release);

      auto pile = get_pile();

      // open cb here

      return 0;
    };
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[0]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec3 position;
      std::istringstream iss(fan::string(text).c_str());
      std::size_t i = 0;
      while (iss >> position[i++]) { iss.ignore(); }

      pile->loco.text.set_position(&instance->cid, position);
      pile->loco.text.set_depth(&instance->cid, position.z);

      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));

    f32_t size = pile->loco.text.get_font_size(&instance->cid);
    p.text = fan::format("{:.2f}", size);
    p.text_value = "";
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[1]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      f32_t size;
      std::istringstream iss(fan::string(text).c_str());
      std::size_t i = 0;
      while (iss >> size) { iss.ignore(); }

      pile->loco.text.set_font_size(&instance->cid, size);

      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));

    p.text = pile->loco.text.get_instance(&instance->cid).text;
    p.text_value = "";
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[2]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      pile->loco.text.set_text(&instance->cid, text);

      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));

    //
    //pile->stage_maker.fgm.button_menu.clear();
    //
    //properties_menu_t::properties_t menup;
    //menup.text = "position";
    //menup.text_value = pile->loco.button.get_button(instance, &loco_t::button_t::instance_t::position).to_string();
    //pile->stage_maker.fgm.button_menu.push_back(menup);
  }

  void push_back(properties_t& p) {
    p.position.z = 1;
    pile_t* pile = get_pile();
    instances.resize(instances.size() + 1);
    uint32_t i = instances.size() - 1;
    instances[i] = new instance_t;
    instances[i]->shape = loco_t::shape_type_t::text;

    if (p.id.empty()) {
      static uint32_t id = 0;
      while (does_id_exist(this, std::to_string(id))) { ++id; }
      instances[i]->id = std::to_string(id);
    }
    else {
      instances[i]->id = p.id;
    }

    loco_t::vfi_t::properties_t vfip;
    vfip.mouse_button_cb = [pile, this, instance = instances[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
      switch (ii_d.button) {
      case fan::mouse_scroll_up: {
        if (pile->stage_maker.fgm.action_flag & action::move) {
          fan::vec3 ps = pile->loco.text.get_instance(&instance->cid).position;
          ps.z += 0.5;
          pile->loco.text.set_position(&instance->cid, ps);
          pile->loco.text.set_depth(&instance->cid, ps.z);
          open_properties(instance);
        }
        return 0;
      }
      case fan::mouse_scroll_down: {
        if (pile->stage_maker.fgm.action_flag & action::move) {
          fan::vec3 ps = pile->loco.text.get_instance(&instance->cid).position;
          ps.z -= 0.5;
          ps.z = fan::clamp((f32_t)ps.z, (f32_t)0.f, (f32_t)ps.z);
          pile->loco.text.set_position(&instance->cid, ps);
          pile->loco.text.set_depth(&instance->cid, ps.z);
          pile->stage_maker.fgm.text.open_properties(instance);
        }
        return 0;
      }
      case fan::mouse_left: {
        break;
      }
      default: {
        return 0;
      }
      }
      if (ii_d.button_state == fan::mouse_state::press) {
        ii_d.flag->ignore_move_focus_check = true;
        ii_d.vfi->set_focus_keyboard(ii_d.vfi->get_focus_mouse());
      }
      if (ii_d.button_state == fan::mouse_state::release) {
        ii_d.flag->ignore_move_focus_check = false;
      }
      pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);

      if (ii_d.button_state == fan::mouse_state::release) {
        pile->stage_maker.fgm.text.release();
        // TODO FIX, erases in near bottom
        if (!pile->stage_maker.fgm.viewport[viewport_area::editor].inside(pile->loco.get_mouse_position()) && !holding_special_key) {
          pile->stage_maker.fgm.text.erase(&instance->cid);
        }
        return 0;
      }
      if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
        return 0;
      }
      pile->stage_maker.fgm.action_flag |= action::move;
      pile->stage_maker.fgm.click_position = ii_d.position;
      pile->stage_maker.fgm.move_offset = fan::vec2(pile->loco.text.get_instance(&instance->cid).position) - pile->stage_maker.fgm.click_position;
      pile->stage_maker.fgm.resize_offset = pile->stage_maker.fgm.click_position;
      fan::vec3 rp = pile->loco.text.get_instance(&instance->cid).position;
      f32_t rs = pile->loco.text.get_font_size(&instance->cid);
      pile->stage_maker.fgm.resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
      pile->stage_maker.fgm.text.open_properties(instance);
      return 0;
    };
    vfip.mouse_move_cb = [this, i](const loco_t::mouse_move_data_t& ii_d) -> int {
      if (ii_d.flag->ignore_move_focus_check == false) {
        return 0;
      }

      pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
      instance_t* instance = pile->stage_maker.fgm.text.instances[i];
      if (holding_special_key) {
        fan::vec3 ps = pile->loco.text.get_instance(&instance->cid).position;
        f32_t rs = pile->loco.text.get_font_size(&instance->cid);

        static constexpr f32_t minimum_rectangle_size = 0.03;
        static constexpr fan::vec2i multiplier[] = { {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };

        rs += ((ii_d.position - pile->stage_maker.fgm.resize_offset) * multiplier[pile->stage_maker.fgm.resize_side] / 2).x;

        if (rs == minimum_rectangle_size) {
          pile->stage_maker.fgm.resize_offset = ii_d.position;
        }

        bool ret = 0;
        if (rs < minimum_rectangle_size) {
          rs = minimum_rectangle_size;
          if (!(rs < minimum_rectangle_size)) {
            ps.x += (ii_d.position.x - pile->stage_maker.fgm.resize_offset.x) / 2;
            pile->stage_maker.fgm.resize_offset.x = ii_d.position.x;
          }
          ret = 1;
        }

        if (rs != minimum_rectangle_size) {
          ps += (ii_d.position - pile->stage_maker.fgm.resize_offset) / 2;
        }
        if (rs == minimum_rectangle_size) {
          ps = pile->loco.text.get_instance(&instance->cid).position;
        }

        pile->stage_maker.fgm.text.set_font_size(instance, rs);
        pile->stage_maker.fgm.text.set_position(instance, ps);
        pile->stage_maker.fgm.text.open_properties(instance);

        if (ret) {
          return 0;
        }

        pile->stage_maker.fgm.resize_offset = ii_d.position;
        pile->stage_maker.fgm.move_offset = fan::vec2(ps) - ii_d.position;
        return 0;
      }

      fan::vec3 ps = pile->loco.text.get_instance(&instance->cid).position;
      fan::vec3 p;
      p.x = ii_d.position.x + pile->stage_maker.fgm.move_offset.x;
      p.y = ii_d.position.y + pile->stage_maker.fgm.move_offset.y;
      p.z = ps.z;
      pile->stage_maker.fgm.text.set_position(instance, p);
      pile->stage_maker.fgm.text.open_properties(instance);

      return 0;
    };
    vfip.keyboard_cb = [this, i](const loco_t::keyboard_data_t& kd) -> int {
      pile_t* pile = OFFSETLESS(OFFSETLESS(kd.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
      auto* instance = pile->stage_maker.fgm.text.instances[i];
      switch (kd.key) {
      case fan::key_delete: {
        if (kd.keyboard_state != fan::keyboard_state::press) {
          return 0;
        }
        switch (kd.keyboard_state) {
        case fan::keyboard_state::press: {
          pile->stage_maker.fgm.text.erase(&pile->stage_maker.fgm.text.instances[i]->cid);
          pile->stage_maker.fgm.invalidate_focus();
          break;
        }
        }
        break;
      }
      case fan::key_c: {
        pile->stage_maker.fgm.text.holding_special_key = kd.keyboard_state == fan::keyboard_state::release ? 0 : 1;
        break;
      }
      case fan::key_left:
      case fan::key_right:
      case fan::key_up:
      case fan::key_down:
      {
        if (kd.keyboard_state == fan::keyboard_state::release) {
          return 0;
        }
        if (kd.key == fan::key_left) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(-1, 0));
        if (kd.key == fan::key_right) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(1, 0));
        if (kd.key == fan::key_up) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(0, -1));
        if (kd.key == fan::key_down) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(0, 1));
        open_properties(instance);
        break;
      }
      #undef something_long
      }
      return 0;
    };
    pile->loco.text.push_back(&instances[i]->cid, p);
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = pile->loco.text.get_text_size(&instances[i]->cid);
    vfip.shape.rectangle.matrices = p.matrices;
    vfip.shape.rectangle.viewport = p.viewport;
    pile->loco.push_back_input_hitbox(&instances[i]->vfi_id, vfip);
  }
  void erase(fan::graphics::cid_t* cid) {
    loco_t& loco = *get_loco();
    close_properties();
    loco.text.erase(cid);
    for (uint32_t i = 0; i < instances.size(); i++) {
      if (&instances[i]->cid == cid) {
        instances.erase(instances.begin() + i);
        break;
      }
    }
    release();
  }

  fgm_make_clear_f(
    pile->loco.text.erase(&it->cid);
    pile->loco.vfi.erase(&it->vfi_id);
  );

  fan::vec3 get_position(instance_t* instance) {
    auto pile = get_pile();
    return pile->loco.text.get_instance(&instance->cid).position;
  }

  void set_position(instance_t* instance, const fan::vec3& position) {
    auto pile = get_pile();
    pile->loco.text.set_position(&instance->cid, position);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::position, position);
  }
  void set_font_size(instance_t* instance, f32_t font_size) {
    auto pile = get_pile();
    pile->loco.text.set_font_size(&instance->cid, font_size);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::size, pile->loco.text.get_text_size(&instance->cid));
  }

  fan::string to_string() const {
    auto* loco = get_loco();
    fan::string f;
    uint32_t instances_count = instances.size();
    fan::write_to_string(f, loco_t::shape_type_t::text);
    fan::write_to_string(f, instances_count);
    for (auto it : instances) {

      stage_maker_shape_format::shape_text_t data;
      data.position = loco->text.get_instance(&it->cid).position;
      data.size = loco->text.get_instance(&it->cid).font_size;
      data.text = loco->text.get_instance(&it->cid).text;
      data.id = it->id;
      f += shape_to_string(data);
    }
    return f;
  }

  uint64_t from_string(const fan::string& f) {
    auto* pile = get_pile();
    uint64_t off = 0;
    loco_t::shape_type_t::_t shape_type = fan::read_data<loco_t::shape_type_t::_t>(f, off);
    if (shape_type != loco_t::shape_type_t::fgm_shape_loco_name) {
      return 0;
    }
    uint32_t instance_count = fan::read_data<uint32_t>(f, off);
    for (uint32_t i = 0; i < instance_count; ++i) {
      stage_maker_shape_format::shape_text_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
        });
      text_t::properties_t p;
      p.position = data.position;
      p.font_size = data.size;
      p.text = data.text;
      p.matrices = &pile->stage_maker.fgm.matrices[viewport_area::editor];
      p.viewport = &pile->stage_maker.fgm.viewport[viewport_area::editor];
      p.id = data.id;
      push_back(p);
    }
    return off;
  }

  #include "fgm_shape_builder_close.h"
};

struct hitbox_t {

  static constexpr const char* cb_names[] = { "mouse_button","mouse_move", "keyboard", "text" };

  struct properties_t : loco_t::sprite_t::properties_t{
    loco_t::vfi_t::shape_type_t vfi_type;
    fan::string id;
  };

  uint8_t holding_special_key = 0;

  #define fgm_shape_name hitbox
  #define fgm_shape_manual_properties
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    loco_t::vfi_t::shape_id_t vfi_id; \
    uint16_t shape; \
    fan::string id; \
    loco_t::vfi_t::shape_type_t shape_type;
  #include "fgm_shape_builder.h"

  void close_properties() {
    auto pile = get_pile();
    if (pile->stage_maker.fgm.properties_open) {
      pile->stage_maker.fgm.properties_open = false;
      pile->stage_maker.fgm.properties_nrs.clear();
      pile->stage_maker.fgm.text_box_menu.erase(pile->stage_maker.fgm.properties_nr);
    }
  }

  void open_properties(instance_t* instance) {
    auto pile = get_pile();

    close_properties();
    pile->stage_maker.fgm.properties_open = true;

    text_box_menu_t::open_properties_t menup;
    menup.matrices = &pile->stage_maker.fgm.matrices[pile_t::stage_maker_t::fgm_t::viewport_area::properties];
    menup.viewport = &pile->stage_maker.fgm.viewport[pile_t::stage_maker_t::fgm_t::viewport_area::properties];
    menup.theme = &pile->stage_maker.fgm.theme;
    menup.position = fan::vec2(0, -0.8);
    menup.gui_size = 0.08;
    auto nr = pile->stage_maker.fgm.text_box_menu.push_menu(menup);
    pile->stage_maker.fgm.properties_nr = nr;
    text_box_menu_t::properties_t p;
    auto position = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
    p.text = fan::format("{:.2f}, {:.2f}, {:.2f}", position.x, position.y, position.z);
    p.text_value = "add cbs";
    p.mouse_button_cb = [this, instance](const loco_t::mouse_button_data_t& mb) -> int {
      use_key_lambda(fan::mouse_left, fan::mouse_state::release);

      auto pile = get_pile();

      // open cb here

      return 0;
    };
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[0]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec3 position;
      std::istringstream iss(fan::string(text).c_str());
      std::size_t i = 0;
      while (iss >> position[i++]) { iss.ignore(); }

      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
      pile->loco.sprite.sb_set_depth(&instance->cid, position.z);
      pile->loco.vfi.shape_list[instance->vfi_id].shape_data.depth = position.z;

      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));

    auto size = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
    p.text = fan::format("{:.2f}, {:.2f}", size.x, size.y);
    p.text_value = "";
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[1]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      fan::vec2 size;
      std::istringstream iss(fan::string(text).c_str());
      std::size_t i = 0;
      while (iss >> size[i++]) { iss.ignore(); }

      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);


      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));

    p.text = fan::to_string(instance->shape_type);
    p.text_value = "";
    p.keyboard_cb = [pile, this, instance, nr](const loco_t::keyboard_data_t& d) -> int {
      if (d.key != fan::key_enter) {
        return 0;
      }
      if (d.keyboard_state != fan::keyboard_state::press) {
        return 0;
      }

      auto& it = pile->loco.menu_maker_text_box.instances[nr].base.instances[pile->stage_maker.fgm.properties_nrs[2]];
      auto text = pile->loco.text_box.get_text(&it.cid);

      uint32_t shape_type;
      std::istringstream iss(fan::string(text).c_str());
      while (iss >> shape_type) { iss.ignore(); }

      instance->shape_type = shape_type;

      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));
  }
  void push_back(properties_t& p) {
    p.position.z = 1;
    pile_t* pile = get_pile();
    shape_builder_push_back
    instances[i]->shape_type = p.vfi_type;

    if (p.id.empty()) {
      static uint32_t id = 0;
      while (does_id_exist(this, std::to_string(id))) { ++id; }
      instances[i]->id = std::to_string(id);
    }
    else {
      instances[i]->id = p.id;
    }

    loco_t::vfi_t::properties_t vfip;
    vfip.mouse_button_cb = [pile, this, instance = instances[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
      switch (ii_d.button) {
      case fan::mouse_scroll_up: {
        if (pile->stage_maker.fgm.action_flag & action::move) {
          fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
          ps.z += 0.5;
          pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
          pile->loco.sprite.sb_set_depth(&instance->cid, ps.z);
          open_properties(instance);
        }
        return 0;
      }
      case fan::mouse_scroll_down: {
        if (pile->stage_maker.fgm.action_flag & action::move) {
          fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
          ps.z -= 0.5;
          ps.z = fan::clamp((f32_t)ps.z, (f32_t)0.f, (f32_t)ps.z);
          pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
          pile->loco.sprite.sb_set_depth(&instance->cid, ps.z);
          open_properties(instance);
        }
        return 0;
      }
      case fan::mouse_left: {
        break;
      }
      default: {
        return 0;
      }
      }
      if (ii_d.button_state == fan::mouse_state::press) {
        ii_d.flag->ignore_move_focus_check = true;
        ii_d.vfi->set_focus_keyboard(ii_d.vfi->get_focus_mouse());
      }
      if (ii_d.button_state == fan::mouse_state::release) {
        ii_d.flag->ignore_move_focus_check = false;
      }
      pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);

      if (ii_d.button_state == fan::mouse_state::release) {
        release();
        // TODO FIX, erases in near bottom
        if (!pile->stage_maker.fgm.viewport[viewport_area::editor].inside(pile->loco.get_mouse_position()) && !holding_special_key) {
          erase(instance);
        }
        return 0;
      }
      if (ii_d.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
        return 0;
      }
      pile->stage_maker.fgm.action_flag |= action::move;
      auto viewport = pile->loco.sprite.get_viewport(&instance->cid);
      pile->stage_maker.fgm.click_position = ii_d.position;
      pile->stage_maker.fgm.move_offset = fan::vec2(pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position)) - pile->stage_maker.fgm.click_position;
      pile->stage_maker.fgm.resize_offset = pile->stage_maker.fgm.click_position;
      fan::vec3 rp = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
      fan::vec3 rs = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
      pile->stage_maker.fgm.resize_side = fan_2d::collision::rectangle::get_side_collision(ii_d.position, rp, rs);
      open_properties(instance);
      return 0;
    };
    vfip.mouse_move_cb = [this, instance = instances[i]](const loco_t::mouse_move_data_t& ii_d) -> int {
        create_shape_move_resize
        return 0;
    };
    vfip.keyboard_cb = [this, instance = instances[i]](const loco_t::keyboard_data_t& kd) -> int {
      pile_t* pile = OFFSETLESS(OFFSETLESS(kd.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
      switch (kd.key) {
      case fan::key_delete: {
        if (kd.keyboard_state != fan::keyboard_state::press) {
          return 0;
        }
        switch (kd.keyboard_state) {
        case fan::keyboard_state::press: {
          erase(instance);
          pile->stage_maker.fgm.invalidate_focus();
          break;
        }
        }
        break;
      }
      case fan::key_c: {
        holding_special_key = kd.keyboard_state == fan::keyboard_state::release ? 0 : 1;
        break;
      }
      case fan::key_left:
      case fan::key_right:
      case fan::key_up:
      case fan::key_down:
      {
        if (kd.keyboard_state == fan::keyboard_state::release) {
          return 0;
        }
        if (kd.key == fan::key_left) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(-1, 0));
        if (kd.key == fan::key_right) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(1, 0));
        if (kd.key == fan::key_up) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(0, -1));
        if (kd.key == fan::key_down) pile->stage_maker.fgm.move_shape(this, instance, fan::vec2(0, 1));
        open_properties(instance);
        break;
      }
      }
      return 0;
    };
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = p.size;
    vfip.shape.rectangle.matrices = p.matrices;
    vfip.shape.rectangle.viewport = p.viewport;
    pile->loco.push_back_input_hitbox(&instances[i]->vfi_id, vfip);
    pile->loco.sprite.push_back(&instances[i]->cid, p);
  }
  // erases even the code generated by fgm
  void erase(instance_t* instance) {
    auto& pile = *get_pile();

    close_properties();

    auto stage_name = pile.stage_maker.get_selected_name(
      pile.stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id
    );
    auto file_name = pile.stage_maker.get_file_fullpath(stage_name);

    pile.stage_maker.fgm.erase_cbs(&pile, file_name, stage_name, "hitbox", instance, cb_names);

    pile.loco.sprite.erase(&instance->cid);

    for (uint32_t i = 0; i < instances.size(); i++) {
      if (&instances[i]->cid == &instance->cid) {
        instances.erase(instances.begin() + i);
        break;
      }
    }
    release();
  }

  fgm_make_clear_f(
    pile->loco.sprite.erase(&it->cid);
    pile->loco.vfi.erase(&it->vfi_id);
  );

  fan::vec3 get_position(instance_t* instance) {
    auto pile = get_pile();
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
  }
  void set_position(instance_t* instance, const fan::vec3& position) {
    auto pile = get_pile();
    pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::position, position);
  }

  fan::vec2 get_size(instance_t* instance) {
    auto pile = get_pile();
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);
  }
  void set_size(instance_t* instance, const fan::vec2& size) {
    auto pile = get_pile();
    pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);
    pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::size, size);
  }

  fan::string to_string() const {
    auto* loco = get_loco();
    fan::string f;
    uint32_t instances_count = instances.size();
    fan::write_to_string(f, loco_t::shape_type_t::hitbox);
    fan::write_to_string(f, instances_count);
    for (auto it : instances) {
      stage_maker_shape_format::shape_hitbox_t data;
      auto& shape = loco->vfi.shape_list[it->vfi_id];
      auto& shape_data = shape.shape_data;
      switch (it->shape_type) {
        case loco_t::vfi_t::shape_t::always: {
          data.position = fan::vec3(fan::vec2(loco->sprite.get(&it->cid, &loco_t::sprite_t::vi_t::position)), shape_data.depth);
          data.size = loco->sprite.get(&it->cid, &loco_t::sprite_t::vi_t::size);
          break;
        }
        case loco_t::vfi_t::shape_t::rectangle: {
          data.position = loco->sprite.get(&it->cid, &loco_t::sprite_t::vi_t::position);
          data.size = loco->sprite.get(&it->cid, &loco_t::sprite_t::vi_t::size);
        }
      }
      data.id = it->id;
      data.vfi_type = it->shape_type;
      f += shape_to_string(data);
    }
    return f;
  }

  uint64_t from_string(const fan::string& f) {
    auto* pile = get_pile();
    uint64_t off = 0;
    loco_t::shape_type_t::_t shape_type = fan::read_data<loco_t::shape_type_t::_t>(f, off);
    if (shape_type != loco_t::shape_type_t::fgm_shape_loco_name) {
      return 0;
    }
    uint32_t instance_count = fan::read_data<uint32_t>(f, off);
    for (uint32_t i = 0; i < instance_count; ++i) {
      stage_maker_shape_format::shape_hitbox_t data;
      data.iterate_masterpiece([&f, &off](auto& o) {
        o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, off);
      });
      hitbox_t::properties_t sp;
      sp.position = data.position;
      sp.size = data.size;
      sp.image = &pile->stage_maker.fgm.hitbox_image;
      sp.matrices = &pile->stage_maker.fgm.matrices[viewport_area::editor];
      sp.viewport = &pile->stage_maker.fgm.viewport[viewport_area::editor];
      sp.vfi_type = data.vfi_type;
      sp.id = data.id;
      push_back(sp);
    }
    return off;
  }

  #include "fgm_shape_builder_close.h"
};

fan_masterpiece_make(
  (button_t) button,
  (sprite_t) sprite,
  (text_t) text,
  (hitbox_t) hitbox
);

struct button_menu_t {
	
  #define fgm_dont_init_shape
  #define fgm_no_gui_properties
  #define fgm_shape_name button_menu
  #define fgm_shape_loco_name menu_maker_button
  #define fgm_shape_instance_data \
    loco_t::menu_maker_button_t::instance_NodeReference_t nr; \
    std::vector<loco_t::menu_maker_button_t::base_type_t::instance_t> ids;
  #include "fgm_shape_builder.h"

	using open_properties_t = loco_t::menu_maker_button_t::open_properties_t;

	loco_t::menu_maker_button_t::instance_NodeReference_t push_menu(const open_properties_t& op) {
		auto pile = get_pile();
		shape_builder_push_back
		instances[i]->nr = pile->loco.menu_maker_button.push_menu(op);
		return instances[i]->nr;
	}
	loco_t::menu_maker_button_t::base_type_t::instance_NodeReference_t push_back(loco_t::menu_maker_button_t::instance_NodeReference_t id, const properties_t& properties) {
		auto pile = get_pile();
		return pile->loco.menu_maker_button.instances[id].base.push_back(&pile->loco, properties, id);
	}

	void erase(loco_t::menu_maker_button_t::instance_NodeReference_t id) {
		auto pile = get_pile();
		pile->loco.menu_maker_button.erase_menu(id);
		for (uint32_t i = 0; i < instances.size(); i++) {
			if (id == instances[i]->nr) {
				instances.erase(instances.begin() + i);
				break;
			}
		}
	}

  fgm_make_clear_f(
    pile->loco.menu_maker_button.erase_menu(it->nr);
  );

  #include "fgm_shape_builder_close.h"
}button_menu;

struct text_box_menu_t {

  using type_t = loco_t::menu_maker_text_box_t;

  using open_properties_t = type_t::open_properties_t;

  #define fgm_dont_init_shape
  #define fgm_no_gui_properties
  #define fgm_shape_name text_box_menu
  #define fgm_shape_loco_name menu_maker_text_box
  #define fgm_shape_instance_data \
    type_t::instance_NodeReference_t nr; \
    std::vector<type_t::base_type_t::instance_t> ids;
  #include "fgm_shape_builder.h"

  type_t::instance_NodeReference_t push_menu(const open_properties_t& op) {
    auto pile = get_pile();
    shape_builder_push_back
    instances[i]->nr = pile->loco.menu_maker_text_box.push_menu(op);
    return instances[i]->nr;
  }
  type_t::base_type_t::instance_NodeReference_t push_back(type_t::instance_NodeReference_t id, const properties_t& properties) {
    auto pile = get_pile();
    return pile->loco.menu_maker_text_box.instances[id].base.push_back(&pile->loco, properties, id);
  }

  void erase(type_t::instance_NodeReference_t id) {
    auto pile = get_pile();
    pile->loco.menu_maker_text_box.erase_menu(id);
    for (uint32_t i = 0; i < instances.size(); i++) {
      if (id == instances[i]->nr) {
        delete instances[i];
        instances.erase(instances.begin() + i);
        break;
      }
    }
  }

  fgm_make_clear_f(
    pile->loco.menu_maker_text_box.erase_menu(it->nr);
  );

  #include "fgm_shape_builder_close.h"
}text_box_menu;

std::vector<loco_t::menu_maker_text_box_base_t::instance_NodeReference_t> properties_nrs;
bool properties_open = false;

loco_t::menu_maker_text_box_t::instance_NodeReference_t properties_nr{(uint16_t) - 1};