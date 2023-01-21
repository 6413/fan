#include <fmt/core.h>

struct line_t {

  #define fgm_shape_name line
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    uint16_t shape;
  #include "fgm_shape_builder.h"

	void push_back(properties_t& p) {
		loco_t& loco = *get_loco();
    shape_builder_push_back
		loco.line.push_back(&instance[i]->cid, p);
	}
	void clear() {
		loco_t& loco = *get_loco();
		for (auto& it : instance) {
			loco.line.erase(&it->cid);
			delete it;
		}
		instance.clear();
	}

  #include "fgm_shape_builder_close.h"
}line;

struct global_button_t {

  #define fgm_shape_name global_button
  #define fgm_shape_loco_name button
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    uint16_t shape;
  #include "fgm_shape_builder.h"

	void push_back(properties_t& p) {
		p.position.z = 1;
		loco_t& loco = *get_loco();
		instance.resize(instance.size() + 1);
		uint32_t i = instance.size() - 1;
		instance[i] = new instance_t;
		instance[i]->shape = shapes::button;
		loco.button.push_back(&instance[i]->cid, p);
	}

	void clear() {
		loco_t& loco = *get_loco();
		for (auto& it : instance) {
			loco.button.erase(&it->cid);
			delete it;
		}
		instance.clear();
	}

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

struct button_t {
  uint8_t holding_special_key = 0;
	using properties_t = loco_t::button_t::properties_t;

  #define fgm_shape_name button
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    uint16_t shape; \
  fan_2d::graphics::gui::theme_t theme;
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

	void release() {
		pile_t* pile = get_pile();
		pile->stage_maker.fgm.move_offset = 0;
		pile->stage_maker.fgm.action_flag &= ~action::move;
	}
	void push_back(properties_t& p) {
		p.position.z = 1;
		pile_t* pile = get_pile();
		instance.resize(instance.size() + 1);
		uint32_t i = instance.size() - 1;
		instance[i] = new instance_t;
		instance[i]->shape = shapes::button;
		instance[i]->theme = *pile->loco.get_context()->theme_list[p.theme].theme_id;
		p.mouse_button_cb = [this, instance = instance[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
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
					pile->stage_maker.fgm.button.erase(&instance->cid);
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
		p.mouse_move_cb = [this, instance = instance[i]](const loco_t::mouse_move_data_t& ii_d) -> int {
			if (ii_d.flag->ignore_move_focus_check == false) {
				return 0;
			}
			pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
      if (!(pile->stage_maker.fgm.action_flag & action::move)) {
        return 0;
      }

			if (holding_special_key) {
				fan::vec3 ps = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
				fan::vec3 rs = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::size);

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
					ps = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
				}

				pile->loco.button.set_size(&instance->cid, rs);
				pile->loco.button.set_position(&instance->cid, ps);

				if (ret) {
					return 0;
				}

				pile->stage_maker.fgm.resize_offset = ii_d.position;
				pile->stage_maker.fgm.move_offset = ps - fan::vec3(ii_d.position, 0);
        pile->stage_maker.fgm.button.open_properties(instance);
				return 0;
			}

      fan::vec3 ps = pile->loco.button.get_button(&instance->cid, &loco_t::button_t::vi_t::position);
			fan::vec3 p;
			p.x = ii_d.position.x + pile->stage_maker.fgm.move_offset.x;
			p.y = ii_d.position.y + pile->stage_maker.fgm.move_offset.y;
			p.z = ps.z;
			pile->loco.button.set_position(&instance->cid, p);

      pile->stage_maker.fgm.button.open_properties(instance);

			return 0;
		};
		p.keyboard_cb = [this, instance = instance[i]](const loco_t::keyboard_data_t& kd) -> int {
			pile_t* pile = OFFSETLESS(OFFSETLESS(kd.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);

			switch (kd.key) {
				case fan::key_delete: {
          if (kd.keyboard_state != fan::keyboard_state::press) {
            return 0;
          }
					switch (kd.keyboard_state) {
						case fan::keyboard_state::press: {
							pile->stage_maker.fgm.button.erase(&instance->cid);
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
		pile->loco.button.push_back(&instance[i]->cid, p);
		pile->loco.button.set_theme(&instance[i]->cid, loco_t::button_t::inactive);
		auto builder_cid = &instance[i]->cid;
		auto ri = pile->loco.button.get_ri(builder_cid);
		pile->loco.vfi.set_focus_mouse(ri.vfi_id);
	}
	void erase(fan::graphics::cid_t* cid) {
		pile_t* pile = OFFSETLESS(get_loco(), pile_t, loco_var_name);
		pile->loco.button.erase(cid);
		for (uint32_t i = 0; i < instance.size(); i++) {
			if (&instance[i]->cid == cid) {
				auto stage_name = pile->stage_maker.get_selected_name(
					pile,
					pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id,
					pile->loco.menu_maker_button.get_selected_id(pile->stage_maker.instances[pile_t::stage_maker_t::stage_t::stage_instance].menu_id)
				);
				auto file_name = pile->stage_maker.get_file_fullpath(stage_name);

				fan::string str;
				fan::io::file::read(file_name, &str);
				auto find = fan::format("int button{}_click_cb", fan::to_string(i));
				std::size_t begin = str.find(find) - 2;
				std::size_t end = str.find("}", begin) + 1;
				str.erase(begin, end - begin);
				fan::io::file::write(file_name, str, std::ios_base::binary);
				instance.erase(instance.begin() + i);
				break;
			}
		}
		release();
	}
	void clear() {
    close_properties();
		loco_t& loco = *get_loco();
		for (auto& it : instance) {
			loco.button.erase(&it->cid);
			delete it;
		}
		instance.clear();
	}

  fan::vec3 get_position(instance_t* instance) {
    return get_loco()->button.get(&instance->cid, &loco_t::button_t::vi_t::position);
  }
  void set_position(instance_t* instance, const fan::vec3& position) {
    get_loco()->button.set_position(&instance->cid, position);
  }

  #include "fgm_shape_builder_close.h"
}button;

struct sprite_t {
  struct properties_t : loco_t::sprite_t::properties_t {
    fan::string texturepack_name;
  };
  uint8_t holding_special_key = 0;

  struct instance_t {
    fan::graphics::cid_t cid;
    loco_t::vfi_t::shape_id_t vfi_id;
    uint16_t shape;
    fan::string texturepack_name;
  };

	loco_t* get_loco() {
		return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, sprite), stage_maker_t, fgm))->get_loco();
	}
	pile_t* get_pile() {
		return OFFSETLESS(get_loco(), pile_t, loco_var_name);
	}

  void close_properties() {
    auto pile = get_pile();
    if (pile->stage_maker.fgm.properties_open) {
      pile->stage_maker.fgm.properties_open = false;
      pile->stage_maker.fgm.properties_nrs.clear();
      pile->stage_maker.fgm.text_box_menu.erase(pile->stage_maker.fgm.properties_nr);
    }
  }

	void open_properties(sprite_t::instance_t* instance) {
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
      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::tc_position, ti.position / data.size);
      pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::tc_size, ti.size / data.size);
      instance->texturepack_name = text;
      return 0;
    };
    pile->stage_maker.fgm.properties_nrs.push_back(pile->stage_maker.fgm.text_box_menu.push_back(nr, p));
	}

	void release() {
		pile_t* pile = get_pile();
		pile->stage_maker.fgm.move_offset = 0;
		pile->stage_maker.fgm.action_flag &= ~action::move;
	}
	void push_back(properties_t& p) {
		p.position.z = 1;
		pile_t* pile = get_pile();
		instances.resize(instances.size() + 1);
		uint32_t i = instances.size() - 1;
		instances[i] = new instance_t;
		instances[i]->shape = shapes::sprite;
    instances[i]->texturepack_name = p.texturepack_name;
		loco_t::vfi_t::properties_t vfip;
		vfip.mouse_button_cb = [pile, this, instance = instances[i]](const loco_t::mouse_button_data_t& ii_d) -> int {
      switch (ii_d.button) {
        case fan::mouse_scroll_up: {
          if (pile->stage_maker.fgm.action_flag & action::move) {
            fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
            ps.z += 0.5;
            pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, ps);
            pile->loco.sprite.set_depth(&instance->cid, ps.z);
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
            pile->loco.sprite.set_depth(&instance->cid, ps.z);
            pile->stage_maker.fgm.sprite.open_properties(instance);
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
				pile->stage_maker.fgm.sprite.release();
				// TODO FIX, erases in near bottom
				if (!pile->stage_maker.fgm.viewport[viewport_area::editor].inside(pile->loco.get_mouse_position()) && !holding_special_key) {
					pile->stage_maker.fgm.sprite.erase(&instance->cid);
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
			pile->stage_maker.fgm.sprite.open_properties(instance);
			return 0;
		};
		vfip.mouse_move_cb = [this, i](const loco_t::mouse_move_data_t& ii_d) -> int {
			if (ii_d.flag->ignore_move_focus_check == false) {
				return 0;
			}

			pile_t* pile = OFFSETLESS(OFFSETLESS(ii_d.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
			instance_t* instance = pile->stage_maker.fgm.sprite.instances[i];
			if (holding_special_key) {
				fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
				fan::vec3 rs = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::size);

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
					ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
				}

				pile->stage_maker.fgm.sprite.set_size(instance, rs);
				pile->stage_maker.fgm.sprite.set_position(instance, ps);
        pile->stage_maker.fgm.sprite.open_properties(instance);

				if (ret) {
					return 0;
				}

				pile->stage_maker.fgm.resize_offset = ii_d.position;
				pile->stage_maker.fgm.move_offset = fan::vec2(ps) - ii_d.position;
				return 0;
			}

      fan::vec3 ps = pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
			fan::vec3 p;
			p.x = ii_d.position.x + pile->stage_maker.fgm.move_offset.x;
			p.y = ii_d.position.y + pile->stage_maker.fgm.move_offset.y;
			p.z = ps.z;
			pile->stage_maker.fgm.sprite.set_position(instance, p);
      pile->stage_maker.fgm.sprite.open_properties(instance);

			return 0;
		};
		vfip.keyboard_cb = [this, i](const loco_t::keyboard_data_t& kd) -> int {
			pile_t* pile = OFFSETLESS(OFFSETLESS(kd.vfi, loco_t, vfi_var_name), pile_t, loco_var_name);
      auto* instance = pile->stage_maker.fgm.sprite.instances[i];
			switch (kd.key) {
				case fan::key_delete: {
          if (kd.keyboard_state != fan::keyboard_state::press) {
            return 0;
          }
				switch (kd.keyboard_state) {
					case fan::keyboard_state::press: {
						pile->stage_maker.fgm.sprite.erase(&pile->stage_maker.fgm.sprite.instances[i]->cid);
						pile->stage_maker.fgm.invalidate_focus();
						break;
					}
				}
				break;
			}
			case fan::key_c: {
				pile->stage_maker.fgm.sprite.holding_special_key = kd.keyboard_state == fan::keyboard_state::release ? 0 : 1;
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
		instances[i]->vfi_id = pile->loco.push_back_input_hitbox(vfip);
		pile->loco.sprite.push_back(&instances[i]->cid, p);
		//auto builder_cid = &instance[i]->cid;
		//auto block = pile->loco.sprite.sb_get_block(builder_cid);
		//pile->loco.vfi.set_focus_mouse(block->p[builder_cid->instance_id].vfi_id);
	}
	void erase(fan::graphics::cid_t* cid) {
		loco_t& loco = *get_loco();
		loco.sprite.erase(cid);
		for (uint32_t i = 0; i < instances.size(); i++) {
			if (&instances[i]->cid == cid) {
				instances.erase(instances.begin() + i);
				break;
			}
		}
		release();
	}
	void clear() {
    close_properties();
		loco_t& loco = *get_loco();
		for (auto& it : instances) {
			loco.sprite.erase(&it->cid);
			loco.vfi.erase(it->vfi_id);
			delete it;
		}
		instances.clear();
	}

  fan::vec3 get_position(instance_t* instance) {
    auto pile = get_pile();
    return pile->loco.sprite.get(&instance->cid, &loco_t::sprite_t::vi_t::position);
  }
	void set_position(instance_t* instance, const fan::vec3& position) {
		auto pile = get_pile();
		pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::position, position);
		pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::position, position);
	}
	void set_size(instance_t* instance, const fan::vec2& size) {
		auto pile = get_pile();
		pile->loco.sprite.set(&instance->cid, &loco_t::sprite_t::vi_t::size, size);
		pile->loco.vfi.set_rectangle(instance->vfi_id, &loco_t::vfi_t::shape_data_rectangle_t::size, size);
	}

	std::vector<instance_t*> instances;
}sprite;

struct text_t {
  using properties_t = loco_t::text_t::properties_t;
  uint8_t holding_special_key = 0;

  struct instance_t {
    fan::graphics::cid_t cid;
    loco_t::vfi_t::shape_id_t vfi_id;
    uint16_t shape;
    fan::string texturepack_name;
  };

  loco_t* get_loco() {
    return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, text), stage_maker_t, fgm))->get_loco();
  }
  pile_t* get_pile() {
    return OFFSETLESS(get_loco(), pile_t, loco_var_name);
  }

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

  void release() {
    pile_t* pile = get_pile();
    pile->stage_maker.fgm.move_offset = 0;
    pile->stage_maker.fgm.action_flag &= ~action::move;
  }
  void push_back(properties_t& p) {
    p.position.z = 1;
    pile_t* pile = get_pile();
    instances.resize(instances.size() + 1);
    uint32_t i = instances.size() - 1;
    instances[i] = new instance_t;
    instances[i]->shape = shapes::text;
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
    pile->loco.text.push_back(p, &instances[i]->cid);
    vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
    vfip.shape.rectangle.position = p.position;
    vfip.shape.rectangle.size = pile->loco.text.get_text_size(&instances[i]->cid);
    vfip.shape.rectangle.matrices = p.matrices;
    vfip.shape.rectangle.viewport = p.viewport;
    instances[i]->vfi_id = pile->loco.push_back_input_hitbox(vfip);
  }
  void erase(fan::graphics::cid_t* cid) {
    loco_t& loco = *get_loco();
    loco.text.erase(cid);
    for (uint32_t i = 0; i < instances.size(); i++) {
      if (&instances[i]->cid == cid) {
        instances.erase(instances.begin() + i);
        break;
      }
    }
    release();
  }
  void clear() {
    close_properties();
    loco_t& loco = *get_loco();
    for (auto& it : instances) {
      loco.text.erase(&it->cid);
      loco.vfi.erase(it->vfi_id);
      delete it;
    }
    instances.clear();
  }
  
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

  std::vector<instance_t*> instances;
}text;

struct button_menu_t {
	
	loco_t* get_loco() {
		return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, button_menu), stage_maker_t, fgm))->get_loco();
	}
	pile_t* get_pile() {
		return OFFSETLESS(get_loco(), pile_t, loco_var_name);
	}

	using properties_t = loco_t::menu_maker_button_t::properties_t;
	using open_properties_t = loco_t::menu_maker_button_t::open_properties_t;

	struct instance_t {
		loco_t::menu_maker_button_t::instance_NodeReference_t nr;
		std::vector<loco_t::menu_maker_button_t::base_type_t::instance_t> ids;
	};

	loco_t::menu_maker_button_t::instance_NodeReference_t push_menu(const open_properties_t& op) {
		auto pile = get_pile();
		instance_t in;
		in.nr = pile->loco.menu_maker_button.push_menu(op);
		instance.push_back(in);
		return in.nr;
	}
	loco_t::menu_maker_button_t::base_type_t::instance_NodeReference_t push_back(loco_t::menu_maker_button_t::instance_NodeReference_t id, const properties_t& properties) {
		auto pile = get_pile();
		return pile->loco.menu_maker_button.instances[id].base.push_back(&pile->loco, properties, id);
	}

	void erase(loco_t::menu_maker_button_t::instance_NodeReference_t id) {
		auto pile = get_pile();
		pile->loco.menu_maker_button.erase_menu(id);
		for (uint32_t i = 0; i < instance.size(); i++) {
			if (id == instance[i].nr) {
				instance.erase(instance.begin() + i);
				break;
			}
		}
	}

	void clear() {
		auto pile = get_pile();
		for (auto& it : instance) {
			pile->loco.menu_maker_button.erase_menu(it.nr);
		}
		instance.clear();
	}

	std::vector<instance_t> instance;
}button_menu;

struct text_box_menu_t {

  loco_t* get_loco() {
    return ((stage_maker_t*)OFFSETLESS(OFFSETLESS(this, fgm_t, text_box_menu), stage_maker_t, fgm))->get_loco();
  }
  pile_t* get_pile() {
    return OFFSETLESS(get_loco(), pile_t, loco_var_name);
  }

  using type_t = loco_t::menu_maker_text_box_t;

  using properties_t = type_t::properties_t;
  using open_properties_t = type_t::open_properties_t;

  struct instance_t {
    type_t::instance_NodeReference_t nr;
    std::vector<type_t::base_type_t::instance_t> ids;
  };

  type_t::instance_NodeReference_t push_menu(const open_properties_t& op) {
    auto pile = get_pile();
    instance_t in;
    in.nr = pile->loco.menu_maker_text_box.push_menu(op);
    instance.push_back(in);
    return in.nr;
  }
  type_t::base_type_t::instance_NodeReference_t push_back(type_t::instance_NodeReference_t id, const properties_t& properties) {
    auto pile = get_pile();
    return pile->loco.menu_maker_text_box.instances[id].base.push_back(&pile->loco, properties, id);
  }

  void erase(type_t::instance_NodeReference_t id) {
    auto pile = get_pile();
    pile->loco.menu_maker_text_box.erase_menu(id);
    for (uint32_t i = 0; i < instance.size(); i++) {
      if (id == instance[i].nr) {
        instance.erase(instance.begin() + i);
        break;
      }
    }
  }

  void clear() {
    auto pile = get_pile();
    for (auto& it : instance) {
      pile->loco.menu_maker_text_box.erase_menu(it.nr);
    }
    instance.clear();
  }

  std::vector<instance_t> instance;
}text_box_menu;

std::vector<loco_t::menu_maker_text_box_base_t::instance_NodeReference_t> properties_nrs;
bool properties_open = false;

loco_t::menu_maker_text_box_t::instance_NodeReference_t properties_nr{(uint16_t) - 1};