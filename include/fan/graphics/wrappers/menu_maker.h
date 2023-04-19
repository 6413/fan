
#include _FAN_PATH(graphics/gui/menu_maker.h)
struct sb_menu_maker_name {

  using base_type_t = sb_menu_maker_type_name;

	using properties_t = sb_menu_maker_type_name::properties_t;
	using open_properties_t = sb_menu_maker_type_name::open_properties_t;

	using select_data_t = sb_menu_maker_type_name::select_data_t;
	using select_cb_t = fan::function_t<int(const sb_menu_maker_type_name::select_data_t&)>;

	#define BLL_set_AreWeInsideStruct 1
	#define BLL_set_CPP_ConstructDestruct
	#define BLL_set_CPP_Node_ConstructDestruct
	#define BLL_set_BaseLibrary 1
	#define BLL_set_prefix instance
	#define BLL_set_type_node uint16_t
	#define BLL_set_NodeData \
  sb_menu_maker_type_name base; \
  select_cb_t select_cb;
	#define BLL_set_Link 1
	#define BLL_set_StoreFormat 1
	#include _FAN_PATH(BLL/BLL.h)

	using nr_t = instance_NodeReference_t;
	using id_t = sb_menu_maker_type_name::instance_NodeReference_t;

	loco_t* get_loco() {
		loco_t* loco = OFFSETLESS(this, loco_t, sb_menu_maker_var_name);
		return loco;
	}

  sb_menu_maker_name() {

	}

	id_t get_instance_id(nr_t id, loco_t::cid_nt_t& nr) {
		auto it = instances[id].base.instances.GetNodeFirst();
		while (it != instances[id].base.instances.dst) {
			if (instances[id].base.instances[it].id == nr) {
				return it;
			}
		}
		fan::throw_error("failed to find instance id (corruption (gl))");
		return{};
	}

	instance_NodeReference_t push_menu(const open_properties_t& op) {
		auto nr = instances.NewNodeLast();
		instances[nr].select_cb = op.select_cb;
		instances[nr].base.open(get_loco(), op);
		return nr;
	}

	void erase_button_soft(instance_NodeReference_t nr, sb_menu_maker_type_name::instance_NodeReference_t id) {
		if (id == instances[nr].base.selected_id) {
      invalidate_selected(nr);
		}
		instances[nr].base.erase_soft(get_loco(), id);
	}
	void erase_button(instance_NodeReference_t nr, sb_menu_maker_type_name::instance_NodeReference_t id) {
		if (id == instances[nr].base.selected_id) {
			invalidate_selected(nr);
		}
		instances[nr].base.erase(get_loco(), id);
	}
	void erase_menu_soft(nr_t nr) {
		instances[nr].base.soft_close(get_loco());
		//instances.Unlink(id);
		//instances.Recycle(id);
	}
	void erase_menu(nr_t nr) {
		instances[nr].base.close(get_loco());
		instances.Unlink(nr);
		instances.Recycle(nr);
	}
	auto push_initialized(nr_t nr, sb_menu_maker_type_name::instance_NodeReference_t id) {
		return instances[nr].base.push_initialized(get_loco(), id, nr);
	}
	auto push_back(nr_t nr, const properties_t& properties) {
		return instances[nr].base.push_back(get_loco(), properties, nr);
	}
  void invalidate_selected(nr_t nr) {
    instances[nr].base.selected.invalidate();
  }
	void set_selected(nr_t nr, loco_t::cid_nt_t& id) {
		instances[nr].base.set_selected(get_loco(), id);
	}
	void set_selected(nr_t nr, sb_menu_maker_type_name::instance_NodeReference_t id) {
		instances[nr].base.set_selected(get_loco(), id);
	}
	auto get_selected_text(nr_t nr) {
		return instances[nr].base.get_selected_text(get_loco());
	}
	loco_t::cid_nt_t& get_selected(nr_t nr) {
		return instances[nr].base.selected;
	}
	sb_menu_maker_type_name::instance_NodeReference_t get_selected_id(nr_t nr) {
		return instances[nr].base.selected_id;
	}
	fan::vec2& get_offset(nr_t nr) {
		return instances[nr].base.global.offset;
	}
	auto size(nr_t nr) {
		return instances[nr].base.instances.Usage();
	}
	bool is_visually_valid(nr_t nr, sb_menu_maker_type_name::instance_NodeReference_t id) {
		return instances[nr].base.is_visually_valid(id);
	}
	void erase_and_update(nr_t nr, sb_menu_maker_type_name::instance_NodeReference_t id) {
		auto loco = get_loco();
		fan::vec2 previous_button_size = loco->sb_menu_maker_shape.get_button(
			&loco->sb_menu_maker_var_name.instances[nr].base.instances[id].id,
			&loco_t::CONCAT(sb_menu_maker_shape, _t)::vi_t::size
		);
		auto it = id;
		it = it.Next(&instances[nr].base.instances);
		erase_button(nr, id);
		instances[nr].base.global.offset.y -= previous_button_size.y * 2;

		while (it != instances[nr].base.instances.dst) {
			auto b_position = loco->sb_menu_maker_shape.get_button(
				instances[nr].base.instances[it].id,
				&loco_t::CONCAT(sb_menu_maker_shape, _t)::vi_t::position
			);
			auto b_size = loco->sb_menu_maker_shape.get_button(
				instances[nr].base.instances[it].id,
				&loco_t::CONCAT(sb_menu_maker_shape, _t)::vi_t::size
			);
			b_position.y -= b_size.y * 2;
			loco->sb_menu_maker_shape.set_position(
				instances[nr].base.instances[it].id,
				b_position
			);
			instances[nr].base.instances[it].position = b_position;
			it = it.Next(&instances[nr].base.instances);
		}
	}

  void set_text(nr_t nr, id_t id, const fan::string& text) {
    instances[nr].base.instances[id].text = text;
    gloco->sb_menu_maker_shape.set_text(&instances[nr].base.instances[id].id, text);
  }

	fan::vec2 get_button_measurements(nr_t nr) {
		return instances[nr].base.get_button_measurements();
	}
	static fan::vec2 get_button_measurements(f32_t gui_size) {
		return sb_menu_maker_type_name::get_button_measurements(gui_size);
	}



	instance_t instances;

}sb_menu_maker_var_name;
#undef sb_menu_maker_var_name
#undef sb_menu_maker_type_name