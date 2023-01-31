#ifndef stage_loader_path
#define stage_loader_path
#endif

#include "common.h"

struct stage_loader_t {

protected:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix stage_list
  #define BLL_set_type_node uint16_t
  #define BLL_set_NodeData \
  loco_t::update_callback_nr_t update_nr; \
  fan::window_t::resize_callback_NodeReference_t resize_nr; \
  void* stage;
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  using nr_t = stage_list_NodeReference_t;
  stage_list_t stage_list;

  struct stage_open_properties_t {
    loco_t::matrices_t* matrices;
    fan::graphics::viewport_t* viewport;
    loco_t::theme_t* theme;

    stage_loader_t::nr_t parent_id;
    uint32_t itToDepthMultiplier = 0x100;
  };

  template <typename T = __empty_struct>
  struct stage_common_t_t {

    stage_common_t_t(auto* loader, auto* loco, const stage_open_properties_t& properties) {
      T* stage = (T*)this;
      stage->stage_id = loader->stage_list.NewNodeLast();
      if (stage->stage_id.Prev(&loader->stage_list) != loader->stage_list.src) {
        stage->it = ((stage_common_t*)&loader->stage_list[stage->stage_id.Prev(&loader->stage_list)])->it + 1;
      }
      else {
        stage->it = 0;
      }
      stage->parent_id = properties.parent_id;
      loader->load_fgm(loco, (T*)this, properties, stage->stage_name);
    }
    void close(auto* loco) {
      T* stage = (T*)this;
      stage->close(*loco);
    }

    nr_t stage_id;
    uint32_t it;

  protected:
    #define BLL_set_CPP_ConstructDestruct
    #define BLL_set_CPP_Node_ConstructDestruct
    #define BLL_set_BaseLibrary 1
    #define BLL_set_prefix cid_list
    #define BLL_set_type_node uint16_t
    #define BLL_set_NodeData \
    uint8_t type; \
    fan::graphics::cid_t cid;
    #define BLL_set_Link 1
    #define BLL_set_StoreFormat 1
    #define BLL_set_AreWeInsideStruct 1
    #define BLL_set_StoreFormat1_ElementPerBlock 0x100
    #include _FAN_PATH(BLL/BLL.h)
  public:

    cid_list_t cid_list;

    stage_loader_t::nr_t parent_id;
  };

  using stage_common_t = stage_common_t_t<>;

	#include _PATH_QUOTE(stage_loader_path/stages_compile/stage.h)

  using key_t = std::pair<void*, fan::string>;

  struct pair_hasher_t {
    std::size_t operator()(const key_t& pair) const {
      return std::hash<decltype(pair.first)>()(pair.first) ^ std::hash<std::string>()(pair.second);
    }
  };

  struct pair_equal_t {
    bool operator()(const key_t& lhs, const key_t& rhs) const {
      return lhs.first == rhs.first && lhs.second == rhs.second;
    }
  };

  using cid_map_t = std::unordered_map<key_t, fan::graphics::cid_t*, pair_hasher_t, pair_equal_t>;
  cid_map_t cid_map;

  fan::graphics::cid_t* get_cid(void* stage_ptr, const fan::string id) {
    auto found = cid_map.find(std::make_pair(stage_ptr, id));
    if (found == cid_map.end()) {
      return nullptr;
    }
    return found->second;
  }

	void open(loco_t* loco, loco_t::texturepack_t* tp) {
    texturepack = tp;
	}
  void close(loco_t* loco) {

  }

  void load_fgm(loco_t* loco, auto* stage, const stage_open_properties_t& op, const char* stage_name) {

    fan::string full_path = fan::string("stages_runtime/") + stage_name + ".fgm";
    fan::string f;
    fan::io::file::read(full_path, &f);
    uint64_t off = 0;

    while (off < f.size()) {
      auto shape_type = fan::io::file::read_data<loco_t::shape_type_t::_t>(f, off);
      uint32_t instance_count = fan::io::file::read_data<uint32_t>(f, off);

      for (uint32_t i = 0; i < instance_count; ++i) {
        auto nr = stage->cid_list.NewNodeLast();
        switch (shape_type) {
				  case loco_t::shape_type_t::button: {
          auto data = fan::io::file::read_data<stage_maker_shape_format::shape_button_t>(f, off);
          auto text = fan::io::file::read_data<fan::string>(f, off);
          loco_t::button_t::properties_t bp;
          bp.position = data.position;
          bp.position.z += stage->it * op.itToDepthMultiplier;
          bp.size = data.size;
          bp.font_size = data.font_size;
          bp.text = text;
          bp.theme = op.theme;
          bp.matrices = op.matrices;
					bp.viewport = op.viewport;
				  bp.mouse_button_cb = [stage, i](const loco_t::mouse_button_data_t&d) {
					  return (stage->*(stage->button_mouse_button_cb_table[i]))(d);
				  };
          loco->button.push_back(&stage->cid_list[nr].cid, bp);
          cid_map[std::make_pair(stage, "button" + std::to_string(data.id))] = &stage->cid_list[nr].cid;
          break;
        }
        case loco_t::shape_type_t::sprite: {
          auto data = fan::io::file::read_data<stage_maker_shape_format::shape_sprite_t>(f, off);
          auto t = fan::io::file::read_data<fan::string>(f, off);
          loco_t::sprite_t::properties_t sp;
          sp.position = data.position;
          sp.position.z += stage->it * op.itToDepthMultiplier;
          sp.size = data.size;
          loco_t::texturepack_t::ti_t ti;
          if (texturepack->qti(t, &ti)) {
            sp.image = &loco->default_texture;
          }
          else {
            auto& pd = texturepack->get_pixel_data(ti.pack_id);
            sp.image = &pd.image;
            sp.tc_position = ti.position / pd.image.size;
            sp.tc_size = ti.size / pd.image.size;
          }
          sp.matrices = op.matrices;
          sp.viewport = op.viewport;
          loco->sprite.push_back(&stage->cid_list[nr].cid, sp);
          break;
        }
        case loco_t::shape_type_t::text: {
          auto data = fan::io::file::read_data<stage_maker_shape_format::shape_text_t>(f, off);
          auto t = fan::io::file::read_data<fan::string>(f, off);
          loco_t::text_t::properties_t p;
          p.matrices = op.matrices;
          p.viewport = op.viewport;

          p.position = data.position;
          p.position.z += stage->it * op.itToDepthMultiplier;
          p.font_size = data.size;
          p.text = t;
          loco->text.push_back(&stage->cid_list[nr].cid, p);
          break;
        }
        case loco_t::shape_type_t::hitbox: {
          auto data = fan::io::file::read_data<stage_maker_shape_format::shape_hitbox_t>(f, off);
          loco_t::vfi_t::properties_t vfip;
          switch (data.shape_type) {
            case loco_t::vfi_t::shape_t::always: {
              vfip.shape_type = loco_t::vfi_t::shape_t::always;
              vfip.shape.always.z = data.position.z;
              break;
            }
            case loco_t::vfi_t::shape_t::rectangle: {
              vfip.shape_type = loco_t::vfi_t::shape_t::rectangle;
              vfip.shape.rectangle.position = data.position;
              vfip.shape.rectangle.size = data.size;
              vfip.shape.rectangle.matrices = op.matrices;
              vfip.shape.rectangle.viewport = op.viewport;
              break;
            }
          }
          vfip.mouse_button_cb = [stage, i](const loco_t::mouse_button_data_t& d) {
            return (stage->*(stage->hitbox_mouse_button_cb_table[i]))(d);
          };
          vfip.mouse_move_cb = [stage, i](const loco_t::mouse_move_data_t& d) {
            return (stage->*(stage->hitbox_mouse_move_cb_table[i]))(d);
          };
          vfip.keyboard_cb = [stage, i](const loco_t::keyboard_data_t& d) {
            return (stage->*(stage->hitbox_keyboard_cb_table[i]))(d);
          };
          vfip.text_cb = [stage, i](const loco_t::text_data_t& d) {
            return (stage->*(stage->hitbox_text_cb_table[i]))(d);
          };
          vfip.ignore_init_move = true;

          loco->push_back_input_hitbox((loco_t::vfi_t::shape_id_t*)&stage->cid_list[nr].cid, vfip);

          cid_map[std::make_pair(stage, "hitbox" + std::to_string(data.id))] = &stage->cid_list[nr].cid;

          break;
        }
        default: {
          fan::throw_error("i cant find what you talk about - fgm");
          break;
        }
        }
        stage->cid_list[nr].type = shape_type;
      }
    }
  }

	template <typename stage_t>
	stage_loader_t::nr_t push_and_open_stage(auto* loco, const stage_open_properties_t& op) {
		auto* stage = new stage_t(this, loco, op);
		stage_list[stage->stage_id].stage = stage;
    stage_list[stage->stage_id].update_nr = loco->m_update_callback.NewNodeLast();
    loco->m_update_callback[stage_list[stage->stage_id].update_nr] = [stage](loco_t* loco) {
      stage->update(*loco);
    };
    stage_list[stage->stage_id].resize_nr = loco->get_window()->add_resize_callback([stage, loco](const auto&) {
      stage->window_resize_callback(*loco); 
      });
    stage->open(*loco);
		return stage->stage_id;
	}
	void erase_stage(auto* loco, nr_t id) {
		//auto loco = get_loco();
  //  //fan::throw_error("todo");
    auto* stage = (stage_common_t*)stage_list[id].stage;
    stage->close(*loco);
		auto it = stage->cid_list.GetNodeFirst();
		while (it != stage->cid_list.dst) {
			auto& node = stage->cid_list[it];
      switch (node.type) {
        case loco_t::shape_type_t::button: {
          loco->button.erase(&node.cid);
          break;
        }
        case loco_t::shape_type_t::sprite: {
          loco->sprite.erase(&node.cid);
          break;
        }
        case loco_t::shape_type_t::text: {
          loco->text.erase(&node.cid);
          break;
        }
        case loco_t::shape_type_t::hitbox: {
          loco->vfi.erase((loco_t::vfi_t::shape_id_t*)&node.cid);
          break;
        }
      }
			it = it.Next(&stage->cid_list);
		}
    loco->m_update_callback.unlrec(stage_list[id].update_nr);
    loco->get_window()->remove_resize_callback(stage_list[id].resize_nr);
    stage_list.unlrec(id);
	}

  loco_t::texturepack_t* texturepack;
};

#undef stage_loader_path