#include <variant>

struct model_loader_t {

  struct private_ {
    #define fgm_build_model_maker
    #include _FAN_PATH(graphics/gui/fgm/common.h)
    #undef fgm_build_model_maker
  };

  struct iterator_t {
    fan_masterpiece_make(
      (private_::stage_maker_shape_format::shape_sprite_t)sprite,
      (private_::stage_maker_shape_format::shape_mark_t)mark
    );
  };

  using sprite_t = private_::stage_maker_shape_format::shape_sprite_t;
  using mark_t = private_::stage_maker_shape_format::shape_mark_t;

  void load(loco_t::texturepack_t* tp, const fan::string& path, auto lambda) {
    fan::string f;

    if (!fan::io::file::exists(path)) {
      return;
    }
    fan::io::file::read(path, &f);

    if (f.empty()) {
      return;
    }

    iterator_t iterator;

    uint64_t offset = 0;
    // read header
    uint32_t header = fan::read_data<uint32_t>(f, offset);
    iterator.iterate_masterpiece([&](auto& d) {
      // read the type
      auto type = fan::read_data<loco_t::shape_type_t::_t>(f, offset);
      uint32_t instance_count = fan::read_data<uint32_t>(f, offset);
      for (uint32_t i = 0; i < instance_count; ++i) {
        d.iterate_masterpiece([&](auto& o) {
          o = fan::read_data<std::remove_reference_t<decltype(o)>>(f, offset);
        });
        lambda(d);
      }
    });
  }
};

struct cm_t {
  struct instance_t {
    std::variant<
      model_loader_t::mark_t,
      model_loader_t::sprite_t
    > type;
  };

  struct model_t {
    std::vector<std::shared_ptr<loco_t::cid_t>> cids;
    std::unordered_map<std::string, instance_t> instances;
  };

  std::unordered_map<uint32_t, model_t> groups;

  void import_from(const char* path, loco_t::texturepack_t* tp) {
    model_loader_t loader;
    loco_t::texturepack_t::ti_t ti;
    loader.load(tp, path, [&](const auto& data) {
      groups[data.group_id].instances[data.id].type = data;
    });
  }
}; 

struct model_list_t {

  using model_id_t = uint64_t;
  std::unordered_map<model_id_t, cm_t*> model_list;


  model_id_t push_model(cm_t* cms) {
    model_id_t model_id = (model_id_t)cms;
    model_list[model_id] = cms;

    iterate(model_id, group_id, [&]<typename T>(auto shape_id, const T & properties) {
      if constexpr (std::is_same_v<T, model_loader_t::mark_t>) {
        loco_t::rectangle_t::properties_t rp;
        rp.camera = &pile->camera;
        rp.viewport = &pile->viewport;
        rp.position = properties.position;
        rp.size = 0.01;
        rp.color = fan::colors::white;
        m.push_shape(model_id, group_id, rp);
      }
    });

    return model_id;
  }
  void erase(model_id_t id, uint32_t group_id) {
    auto& cids = model_list[id]->groups[group_id].cids;
    for (auto& i : cids) {
      loco_var.erase_shape(i.get());
    }
    model_list[id]->groups.erase(group_id);
  }

  void erase(model_id_t id) {
    auto& groups = model_list[id]->groups;
    for (auto it = groups.begin(); it != groups.end(); ) {
      auto& cids = it->second.cids;
      for (auto& i : cids) {
        loco_var.erase_shape(i.get());
      }
      it = groups.erase(it);
    }
  }

  loco_t::cid_t* push_shape(model_id_t model_id, uint32_t group_id, const auto& properties) {
    auto& cids = model_list[model_id]->groups[group_id].cids;
    cids.emplace_back(std::make_shared<loco_t::cid_t>());
    loco_var.push_shape(cids.back().get(), properties);
    return cids.back().get();
  }


  void iterate(model_id_t model_id, auto lambda) {
    for (auto& it : auto it = model_list[model_id]->groups) {
      iterate(model_id, it->second,lambda);
    }
  }

  void iterate(model_id_t model_id, uint32_t group_id, auto lambda) {
    auto it = model_list[model_id]->groups.find(group_id);
    if (it == model_list[model_id]->groups.end()) {
      fan::throw_error("model iterate - invalid group_id");
    }
    for (auto& i : it->second.instances) {
      std::visit([&](auto&& o) {
        lambda(i.first, o);
        }, i.second.type);
    }
  }
};

#undef loco_var