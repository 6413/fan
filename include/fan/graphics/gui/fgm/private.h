#if defined(fgm_build_stage_maker)
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
void erase_cbs(const fan::string& file_name, const fan::string& stage_name, const fan::string& shape_name, auto* instance, const char* const(&cb_names)[N]) {
  fan::string str;
  fan::io::file::read(file_name, &str);

  const fan::string advance_str = fan::format("struct {0}_t", stage_name);
  auto advance_position = get_stage_maker()->stage_h_str.find(
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

    auto src = get_stage_maker()->stage_h_str.find(find_str, advance_position);
    if (src == fan::string::npos) {
      fan::throw_error("corrupted stage.h");
    }
    src += find_str.size();
    auto dst = get_stage_maker()->stage_h_str.find("]", src);
    if (dst == fan::string::npos) {
      fan::throw_error("corrupted stage.h");
    }
    fan::string tt = get_stage_maker()->stage_h_str.substr(src, dst - src);
    int val = std::stoi(tt);

    // prevent 0 sized array
    if (val != 1) {
      val -= 1;
    }
    get_stage_maker()->stage_h_str.replace(src, dst - src, std::to_string(val));

    find_str = fan::format("&{0}_t::{1}{2}_{3}_cb,", stage_name, shape_name, instance->id, cb_names[j]);

    src = get_stage_maker()->stage_h_str.find(find_str, advance_position);
    get_stage_maker()->stage_h_str.erase(src, find_str.size());
  }
  get_stage_maker()->write_stage();
}

#endif

fan::vec3 get_position(auto* shape, auto* instance) {
  return shape->get_position(instance);
}
void set_position(auto* shape, auto* instance, const fan::vec3& p) {
  shape->set_position(instance, p);
}

void move_shape(auto* shape, auto* instance, const fan::vec2& offset) {
  fan::vec3 p = get_position(shape, instance);
  p += fan::vec3(fan::vec2(
    camera->coordinates.right - camera->coordinates.left,
    camera->coordinates.down - camera->coordinates.up
  ) / pile->loco.get_window()->get_size() * offset, 0);
  set_position(shape, instance, p);
}

void erase_shape(auto* shape, auto* instance) {
  shape->erase(instance);
}

static bool does_id_exist(auto* shape, const fan::string& id) {
  for (const auto& it : shape->instances) {
    if (it->id == id) {
      return true;
    }
  }
  return false;
}

struct line_t {

  #define fgm_shape_name line
  #define fgm_no_gui_properties
  #define fgm_shape_non_moveable_or_resizeable
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    uint16_t shape;
  #include "shape_builder.h"

	void push_back(properties_t& p) {
    shape_builder_push_back();
    uint32_t i = instances.size() - 1;
		pile->loco.line.push_back(&instances[i]->cid, p);
	}

  fgm_make_clear_f(
    pile->loco.line.erase(&it->cid);
  );

  #include "shape_builder_close.h"
}line;

struct global_button_t {

  #define fgm_shape_name global_button
  #define fgm_no_gui_properties
  #define fgm_shape_non_moveable_or_resizeable
  #define fgm_shape_loco_name button
  #define fgm_shape_instance_data \
    fan::graphics::cid_t cid; \
    uint16_t shape;
  #include "shape_builder.h"

	void push_back(properties_t& p) {
		instances.resize(instances.size() + 1);
		uint32_t i = instances.size() - 1;
		instances[i] = new instance_t;
		instances[i]->shape = loco_t::shape_type_t::button;
    pile->loco.button.push_back(&instances[i]->cid, p);
	}
  fgm_make_clear_f(
    pile->loco.button.erase(&it->cid);
  );

  #include "shape_builder_close.h"
}global_button;

struct button_menu_t {
	
  #define fgm_dont_init_shape
  #define fgm_no_gui_properties
  #define fgm_shape_non_moveable_or_resizeable
  #define fgm_shape_name button_menu
  #define fgm_shape_loco_name menu_maker_button
  #define fgm_shape_instance_data \
    loco_t::menu_maker_button_t::instance_NodeReference_t nr; \
    std::vector<loco_t::menu_maker_button_t::base_type_t::instance_t> ids;
  #include "shape_builder.h"

	using open_properties_t = loco_t::menu_maker_button_t::open_properties_t;

	loco_t::menu_maker_button_t::instance_NodeReference_t push_menu(const open_properties_t& op) {
    shape_builder_push_back();
    uint32_t i = instances.size() - 1;
		instances[i]->nr = pile->loco.menu_maker_button.push_menu(op);
		return instances[i]->nr;
	}
	loco_t::menu_maker_button_t::base_type_t::instance_NodeReference_t push_back(loco_t::menu_maker_button_t::instance_NodeReference_t id, const properties_t& properties) {
		return pile->loco.menu_maker_button.instances[id].base.push_back(&pile->loco, properties, id);
	}

	void erase(loco_t::menu_maker_button_t::instance_NodeReference_t id) {
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

  #include "shape_builder_close.h"
}button_menu;

struct text_box_menu_t {

  using type_t = loco_t::menu_maker_text_box_t;

  using open_properties_t = type_t::open_properties_t;

  #define fgm_dont_init_shape
  #define fgm_no_gui_properties
  #define fgm_shape_non_moveable_or_resizeable
  #define fgm_shape_name text_box_menu
  #define fgm_shape_loco_name menu_maker_text_box
  #define fgm_shape_instance_data \
    type_t::instance_NodeReference_t nr; \
    std::vector<type_t::base_type_t::instance_t> ids;
  #include "shape_builder.h"

  type_t::instance_NodeReference_t push_menu(const open_properties_t& op) {
    shape_builder_push_back();
    uint32_t i = instances.size() - 1;
    instances[i]->nr = pile->loco.menu_maker_text_box.push_menu(op);
    return instances[i]->nr;
  }
  type_t::base_type_t::instance_NodeReference_t push_back(type_t::instance_NodeReference_t id, const properties_t& properties) {
    return pile->loco.menu_maker_text_box.instances[id].base.push_back(&pile->loco, properties, id);
  }

  void erase(type_t::instance_NodeReference_t id) {
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

  #include "shape_builder_close.h"
}text_box_menu;

std::vector<loco_t::menu_maker_text_box_base_t::instance_NodeReference_t> properties_nrs;
bool properties_open = false;

loco_t::menu_maker_text_box_t::instance_NodeReference_t properties_nr{(uint16_t) - 1};