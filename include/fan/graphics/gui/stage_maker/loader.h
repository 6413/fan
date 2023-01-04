#ifndef stage_loader_path
#define stage_loader_path
#endif

#include "common.h"

struct stage_loader_t {
	loco_t* get_loco() {
    return OFFSETLESS(this, loco_t, stage_loader);
	}

	#include _FAN_PATH(stage_loader_path/stages/stage.h)

	void open(const char* compiled_tp_path) {
    texturepack.open_compiled(get_loco(), compiled_tp_path);
	}
	void close() {
    texturepack.close();
	}

  void load_fgm(auto* stage, const stage_open_properties_t& op, const char* stage_name) {
	  auto loco = get_loco();

    fan::string full_path = fan::string("stages/") + stage_name + ".fgm";
    fan::string f;
    fan::io::file::read(full_path, &f);
    uint64_t off = 0;

    while (off < f.size()) {
      auto shape_type = fan::io::file::read_data<loco_t::stage_maker_shape_format::shape_type_t::_t>(f, off);
      uint32_t instance_count = fan::io::file::read_data<uint32_t>(f, off);

      for (uint32_t i = 0; i < instance_count; ++i) {
        auto nr = stage->cid_list.NewNodeLast();
        switch (shape_type) {
				  case loco_t::stage_maker_shape_format::shape_type_t::button: {
          auto data = fan::io::file::read_data<loco_t::stage_maker_shape_format::shape_button_t>(f, off);
          auto text = fan::io::file::read_data<fan::wstring>(f, off);
          loco_t::button_t::properties_t bp;
          bp.position = data.position;
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
        case loco_t::stage_maker_shape_format::shape_type_t::sprite: {
          auto data = fan::io::file::read_data<loco_t::stage_maker_shape_format::shape_sprite_t>(f, off);
          loco_t::sprite_t::properties_t sp;
          sp.position = data.position;
          sp.size = data.size;
          loco_t::texturepack::ti_t ti;
          if (loco->stage_loader.texturepack.qti("test.webp", &ti)) {
            fan::throw_error("failed to load texture from texturepack");
          }
          auto pd = loco->stage_loader.texturepack.get_pixel_data(ti.pack_id);
          sp.image = &pd.image;
          sp.tc_position = ti.position / pd.size;
          sp.tc_size = ti.size / pd.size;
          sp.matrices = op.matrices;
          sp.viewport = op.viewport;
          loco->sprite.push_back(&stage->cid_list[nr], sp);
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
	loco_t::stage_loader_t::nr_t push_and_open_stage(const stage_open_properties_t& op) {
		stage_t* stage = new stage_t(get_loco(), op);
		auto& stage_list = loco_t::stage_loader_t::stage::stage_list;
    stage->stage_id = stage_list.NewNodeLast();
		stage_list[stage->stage_id] = stage;
		return stage->stage_id;
	}
	void erase_stage(uint32_t id) {
		//auto& stages = loco_t::stage_loader_t::stage::stages;
		//if (id >= stages.size()) {
		//	return;
		//}

		//auto loco = get_loco();
  //  //fan::throw_error("todo");
		////auto it = stages[id]->instances.GetNodeFirst();
		////while (it != stages[id]->instances.dst) {
		////	auto node = stages[id]->instances[it];
		////	loco->button.erase(&node.cid);
		////	it = it.Next(&stages[id]->instances);
		////}
		//delete stages[id];
		//stages.erase(stages.begin() + id);
	}

  loco_t::texturepack texturepack;
};

#undef stage_loader_path