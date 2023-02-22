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

    using value_type_t = stage_common_t_t<T>;

    stage_common_t_t(auto* loader, auto* loco, const stage_open_properties_t& properties) {
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

    uint32_t file_version = fan::read_data<uint32_t>(f, off);

    switch (file_version) {
      case stage_maker_format_version: {
        #include _FAN_PATH(graphics/gui/stage_maker/fgm_user_loader_version/011.h)
        break;
      }
      default: {
        fan::throw_error("invalid version fgm version number", file_version);
        break;
      }
    }
  }

	template <typename stage_t>
	stage_loader_t::nr_t push_and_open_stage(auto* loco, const stage_open_properties_t& op) {
    auto stage = (stage_t*)malloc(sizeof(stage_t));

    stage->stage_id = stage_list.NewNodeLast();
    if (stage->stage_id.Prev(&stage_list) != stage_list.src) {
      stage->it = ((stage_common_t *)stage_list[stage->stage_id.Prev(&stage_list)].stage)->it + 1;
    }
    else {
      stage->it = 0;
    }
    stage->parent_id = op.parent_id;

    std::construct_at(stage, this, loco, op);

    load_fgm(loco, stage, op, stage->stage_name);

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
    //stage->close(*loco);
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