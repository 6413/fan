struct dropdown_t {

	struct open_properties_t : loco_t::menu_maker_button_t::open_properties_t {

	};
	struct properties_t : loco_t::menu_maker_button_t::properties_t {
		std::vector<fan::string> items;
	};

	using nr_t = loco_t::menu_maker_button_t::nr_t;
	using shape_t = loco_t::menu_maker_button_t::shape_t;

	loco_t* get_loco() {
		loco_t* loco = OFFSETLESS(this, loco_t, dropdown);
		return loco;
	}

	void open() {

	}
	void close() {

	}

	void push_always(f32_t depth, loco_t::mouse_button_cb_t cb) {
		auto loco = get_loco();

		loco_t::vfi_t::properties_t vfip;
		vfip.shape_type = loco_t::vfi_t::shape_t::always;
		vfip.shape.always.z = depth;

		vfip.mouse_button_cb = cb;

		loco->vfi.push_back(&vfi_id, vfip);
	}
	void erase_always() {
		auto loco = get_loco();
		loco->vfi.erase(&vfi_id);
		//vfi_id.NRI = -1;
	}

	void erase_dropdown_menu(auto loco, auto nr, auto id) {

		instances[id].m_open = false;

		auto& mn_instances = loco->menu_maker_button.instances[nr].base.instances;
		auto it = mn_instances.GetNodeFirst();
		if (it == mn_instances.dst) {
			return;
		}
		// iterate to second one to not remove first element
		it = it.Next(&mn_instances);

		while (it != mn_instances.dst) {
			mn_instances.StartSafeNext(it);
			auto node = mn_instances.GetNodeByReference(it);
			loco->menu_maker_button.erase_button(nr, it);
			auto& b = loco->menu_maker_button.instances[nr].base;
			b.global.offset.y -= b.get_button_measurements().y * 2;
			it = mn_instances.EndSafeNext();
		}

		erase_always();
	};

	nr_t push_menu(loco_t::menu_maker_button_t::open_properties_t& op) {
		auto loco = get_loco();
		auto nr = loco->menu_maker_button.push_menu(op);
		instance_t in;
		in.nr = nr;
		in.m_open = false;
		instances.push_back(in);
		return nr;
	}
	uint32_t get_i(nr_t nr) {
		for (uint32_t i = 0; i < instances.size(); i++) {
			if (nr == instances[i].nr) {
				return i;
				break;
			}
		}
		return -1;
	}

	shape_t push_back(nr_t nr, properties_t& p) {
		auto loco = get_loco();
		instances[get_i(nr)].items = p.items;
		p.mouse_button_cb = [this, nr](const loco_t::mouse_button_data_t& mb) -> int {
			if (mb.button != fan::mouse_left) {
				return 0;
			}
			if (mb.button_state != fan::mouse_state::release) {
				return 0;
			}
			if (mb.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
				return 0;
			}

			auto loco = get_loco();
      auto temp = mb.id;
			auto click_position = loco->button.get_button(temp, &loco_t::button_t::vi_t::position);

			uint32_t id = get_i(nr);

			instances[id].menu_nr = nr;
			loco_t::menu_maker_button_t::properties_t mp;
			mp.mouse_button_cb = [this, nr, id](const loco_t::mouse_button_data_t& mb) {

				if (mb.button != fan::mouse_left) {
					return 0;
				}
				if (mb.button_state != fan::mouse_state::release) {
					return 0;
				}
				if (mb.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
					return 0;
				}

				auto loco = get_loco();
				auto top_menu = loco->menu_maker_button.instances[nr].base.instances[loco->menu_maker_button.instances[nr].base.instances.GetNodeFirst()].id;
        auto temp = mb.id;
				auto text = loco->button.get_text(temp);
				loco->button.set_text(
					top_menu,
					text
				);

				//loco->menu_maker_button.set_selected(nr, nullptr);

				erase_dropdown_menu(loco, nr, id);

				return 1;
			};

			if (instances[id].m_open) {
				erase_dropdown_menu(loco, nr, id);
				return 0;
			}

			instances[id].m_open = true;


		push_always(click_position.z + 1, [this, loco, nr, id](const mouse_button_data_t& mb) {
			if (mb.button != fan::mouse_left) {
				return 0;
			}
			if (mb.button_state != fan::mouse_state::release) {
				return 0;
			}
			if (mb.mouse_stage != loco_t::vfi_t::mouse_stage_e::inside) {
				return 0;
			}
			erase_dropdown_menu(loco, nr, id);
			});

			for (uint32_t i = 0; i < instances[id].items.size(); i++) {
				mp.text = instances[id].items[i];
				mp.offset.z = click_position.z + 1;
				loco->menu_maker_button.push_back(instances[id].menu_nr, mp);
			}

			return 0;
		};
		return loco->menu_maker_button.push_back(nr, p);
	}

	struct instance_t {
		nr_t nr;
		nr_t menu_nr;
		std::vector<fan::string> items;
		bool m_open;
	};

	std::vector<instance_t> instances;
	vfi_t::shape_id_t vfi_id;
	// fan::opengl::cid_t menu_cid;

}dropdown;