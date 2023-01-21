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
  #define BLL_set_NodeDataType void *
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  using nr_t = stage_list_NodeReference_t;

  struct stage_open_properties_t {
    loco_t::matrices_list_NodeReference_t matrices;
    fan::graphics::viewport_list_NodeReference_t viewport;
    fan::graphics::theme_list_NodeReference_t theme;

    stage_loader_t::nr_t parent_id;
    uint32_t itToDepthMultiplier = 0x100;
  };

  template <typename T = __empty_struct>
  struct stage_common_t_t {

    stage_common_t_t(auto* loader, auto* loco, const stage_open_properties_t& properties) {
      T* stage = (T*)this;
      loader->load_fgm(loco, (T*)this, properties, stage->stage_name);
      stage->open(loco);
    }
    void close(auto* loco) {
      T* stage = (T*)this;
      stage->close(loco);
    }

    nr_t stage_id;
    uint32_t it;

  protected:
    #define BLL_set_CPP_ConstructDestruct
    #define BLL_set_CPP_Node_ConstructDestruct
    #define BLL_set_BaseLibrary 1
    #define BLL_set_prefix cid_list
    #define BLL_set_type_node uint16_t
    #define BLL_set_NodeDataType fan::graphics::cid_t
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

	#include _PATH_QUOTE(stage_loader_path/stages/stage.h)

	void open(loco_t* loco, loco_t::texturepack_t* tp) {
    texturepack = tp;
	}
  void close(loco_t* loco) {

  }

  void load_fgm(loco_t* loco, auto* stage, const stage_open_properties_t& op, const char* stage_name) {

    fan::string full_path = fan::string("stages/") + stage_name + ".fgm";
    fan::string f;
    fan::io::file::read(full_path, &f);
    uint64_t off = 0;

    while (off < f.size()) {
      auto shape_type = fan::io::file::read_data<stage_maker_shape_format::shape_type_t::_t>(f, off);
      uint32_t instance_count = fan::io::file::read_data<uint32_t>(f, off);

      for (uint32_t i = 0; i < instance_count; ++i) {
        auto nr = stage->cid_list.NewNodeLast();
        switch (shape_type) {
				  case stage_maker_shape_format::shape_type_t::button: {
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
					  return (stage->*(stage->button_click_cb_table[i]))(d); 
				  };
				  
          loco->button.push_back(&stage->cid_list[nr], bp);
          break;
        }
        case stage_maker_shape_format::shape_type_t::sprite: {
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
            auto pd = texturepack->get_pixel_data(ti.pack_id);
            sp.image = &pd.image;
            sp.tc_position = ti.position / pd.size;
            sp.tc_size = ti.size / pd.size;
          }
          sp.matrices = op.matrices;
          sp.viewport = op.viewport;
          loco->sprite.push_back(&stage->cid_list[nr], sp);
          break;
        }
        case stage_maker_shape_format::shape_type_t::text: {
          auto data = fan::io::file::read_data<stage_maker_shape_format::shape_text_t>(f, off);
          auto t = fan::io::file::read_data<fan::string>(f, off);
          loco_t::text_t::properties_t p;
          p.matrices = op.matrices;
          p.viewport = op.viewport;

          p.position = data.position;
          p.position.z += stage->it * op.itToDepthMultiplier;
          p.font_size = data.size;
          p.text = t;
          loco->text.push_back(p, &stage->cid_list[nr]);
          break;
        }
        default: {
          fan::throw_error("i cant find what you talk about - fgm");
          break;
        }
        }
      }
    }
  }

	template <typename stage_t>
	stage_loader_t::nr_t push_and_open_stage(auto* loco, const stage_open_properties_t& op) {
		auto* stage = new stage_t(this, loco, op);
		auto& stage_list = stage_loader_t::stage::stage_list;
    stage->stage_id = stage_list.NewNodeLast();

    if (stage->stage_id.Prev(&stage_list) != stage_list.src) {
      stage->it = ((stage_common_t*)stage_list[stage->stage_id.Prev(&stage_list)])->it + 1;
    }
    else {
      stage->it = 0;
    }

		stage_list[stage->stage_id] = stage;
		return stage->stage_id;
	}
	void erase_stage(auto* loco, nr_t id) {
		//auto loco = get_loco();
  //  //fan::throw_error("todo");
    auto& stage_list = stage_loader_t::stage::stage_list;
    auto* stage = (stage_common_t*)stage_list[id];
		auto it = stage->cid_list.GetNodeFirst();
		while (it != stage->cid_list.dst) {
			auto& node = stage->cid_list[it];
			loco->button.erase(&node);
			it = it.Next(&stage->cid_list);
		}
    stage_list.unlrec(id);
	}

  loco_t::texturepack_t* texturepack;
};

#undef stage_loader_path