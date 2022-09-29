#define sb_menu_maker_var_name menu_maker
#define sb_menu_maker_type_name menu_maker_base_t
#include _FAN_PATH(graphics/gui/menu_maker.h)
struct menu_maker_t {
  using properties_t = menu_maker_base_t::properties_t;
  using open_properties_t = menu_maker_base_t::open_properties_t;

  using select_data_t = menu_maker_base_t::select_data_t;
  using select_cb_t = fan::function_t<int(const menu_maker_base_t::select_data_t&)>;

  #define BLL_set_AreWeInsideStruct 1
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix instance
  #define BLL_set_type_node uint16_t
  #define BLL_set_node_data \
  menu_maker_base_t base; \
  select_cb_t select_cb;
  #define BLL_set_Link 1
  #define BLL_set_StoreFormat 1
  #include _FAN_PATH(BLL/BLL.h)

  using nr_t = instance_NodeReference_t;
  using id_t = menu_maker_base_t::instance_NodeReference_t;
  
  loco_t* get_loco() {
		loco_t* loco = OFFSETLESS(this, loco_t, sb_menu_maker_var_name);
		return loco;
	}

  void open() {
	instances.Open();
  }
  void close() {
	instances.Close();
  }

  id_t get_instance_id(nr_t id, fan::opengl::cid_t* cid) {
	auto it = instances[id].base.instances.GetNodeFirst();
	while (it != instances[id].base.instances.dst) {
	  if (&instances[id].base.instances[it].cid == cid) {
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

  void erase_button_soft(instance_NodeReference_t nr, menu_maker_base_t::instance_NodeReference_t id) {
	if (id == instances[nr].base.selected_id) {
	  set_selected(nr, nullptr);
	}
	instances[nr].base.erase_soft(get_loco(), id);
  }
  void erase_button(instance_NodeReference_t nr, menu_maker_base_t::instance_NodeReference_t id) {
	if (id == instances[nr].base.selected_id) {
	  set_selected(nr, nullptr);
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
  auto push_initialized(nr_t nr, menu_maker_base_t::instance_NodeReference_t id) {
	return instances[nr].base.push_initialized(get_loco(), id, nr);
  }
  auto push_back(nr_t nr, const properties_t& properties) {
	return instances[nr].base.push_back(get_loco(), properties, nr);
  }
  void set_selected(nr_t nr, fan::opengl::cid_t* cid) {
	instances[nr].base.set_selected(get_loco(), cid);
  }
  void set_selected(nr_t nr, menu_maker_base_t::instance_NodeReference_t id) {
	instances[nr].base.set_selected(get_loco(), id);
  }
  fan::string get_selected_text(nr_t nr) {
	return instances[nr].base.get_selected_text(get_loco());
  }
  fan::opengl::cid_t* get_selected(nr_t nr) {
	return instances[nr].base.selected;
  }
  menu_maker_base_t::instance_NodeReference_t get_selected_id(nr_t nr) {
	return instances[nr].base.selected_id;
  }
  fan::vec2& get_offset(nr_t nr) {
	return instances[nr].base.global.offset;
  }
  auto size(nr_t nr) {
	return instances[nr].base.instances.usage();
  }
  bool is_visually_valid(nr_t nr, menu_maker_base_t::instance_NodeReference_t id) {
	return instances[nr].base.is_visually_valid(id);
  }
  void erase_and_update(nr_t nr, menu_maker_base_t::instance_NodeReference_t id) {
	auto loco = get_loco();
	fan::vec2 previous_button_size = loco->button.get_button(
	  &loco->menu_maker.instances[nr].base.instances[id].cid, 
	  &loco_t::button_t::instance_t::size
	);
	auto it = id;
	it = it.Next(&instances[nr].base.instances);
	erase_button(nr, id);
	instances[nr].base.global.offset.y -= previous_button_size.y * 2;

	while (it != instances[nr].base.instances.dst) {
	  auto b_position = loco->button.get_button(
		&instances[nr].base.instances[it].cid,
		&loco_t::button_t::instance_t::position
	  );
	  auto b_size = loco->button.get_button(
		&instances[nr].base.instances[it].cid,
		&loco_t::button_t::instance_t::size
	  );
	  b_position.y -= b_size.y * 2;
	  loco->button.set_position(
		&instances[nr].base.instances[it].cid,
		b_position
	  );
	  instances[nr].base.instances[it].position = b_position;
	  it = it.Next(&instances[nr].base.instances);
	}
  }

  fan::vec2 get_button_measurements(nr_t nr) {
	return instances[nr].base.get_button_measurements();
  }

  instance_t instances;

}sb_menu_maker_var_name;
#undef sb_menu_maker_var_name
#undef sb_menu_maker_type_name