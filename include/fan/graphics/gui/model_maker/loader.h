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

  using sprite_t = model_loader_t::private_::stage_maker_shape_format::shape_sprite_t;
  using mark_t = model_loader_t::private_::stage_maker_shape_format::shape_mark_t;

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

struct model_list_t {

  struct cm_t {
    struct instance_t {
      std::variant<
        model_loader_t::private_::stage_maker_shape_format::shape_mark_t,
        model_loader_t::private_::stage_maker_shape_format::shape_sprite_t
      > type;
    };

    void import_from(const fan::string& path, loco_t::texturepack_t* tp) {
      model_loader_t loader;
      loco_t::texturepack_t::ti_t ti;
      loader.load(tp, path, [&](const auto& data) {
        instances[data.id].type = data;
        });
    }
    std::unordered_map<std::string, instance_t> instances;
  };

  // compiled model
  struct model_id_data_t {
    template <typename T>
    struct id_t {
      using type_t = T;
      type_t internal_;
      std::shared_ptr<loco_t::cid_t> cid;
      fan::vec3 position = 0;
      fan::vec2 size = 0;
      f32_t angle = 0;
      fan::string id;
    };

    struct model_t {
      std::vector<id_t<
        std::variant<
        loco_t::sprite_t*
        #if defined(loco_rectangle)
        , loco_t::rectangle_t*
        #endif
        #if defined(loco_button)
        , loco_t::button_t*
        #endif
        >
        >> cids;
      fan::vec3 position = 0;
      fan::vec2 size = 0;
      f32_t angle = 0;
    };

    std::unordered_map<uint32_t, model_t> groups;

    cm_t cm;

    fan::vec3 position = 0;
    fan::vec2 size = 0;
    f32_t angle = 0;
  };

private:
  //struct mark_t {
  //  fan::vec2 Offset;
  //};
  //struct sprite_t {
  //  cid_t cid;

  //  fan::vec2 Offset;

  //  // excludes viewport, camera, position, ...
  //  sprite_properties_t internal_properties;
  //};
  //struct light_t {
  //  cid_t cid;

  //  fan::vec2 Offset;

  //  // excludes viewport, camera, position, ...
  //  light_properties_t internal_properties;
  //};
  //struct a_group_t{
  //  std::vector<mark_t> MarkList;
  //  std::vector<sprite_t> SpriteList;
  //  std::vector<light_t> LightList;
  //};
  //struct model_id_data_t {
  //  std::unordered_map<uint32_t, a_group_t> GroupList;
  //  fan::vec2 Position;
  //  fan::vec2 Angle;
  //};

  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix model_id_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType model_id_data_t
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  using model_id_t = model_id_list_NodeReference_t;
  model_id_list_t m_model_list;

  struct properties_t {
    loco_t::camera_t* camera = 0;
    fan::graphics::viewport_t* viewport = 0;
  };

  model_id_t push_model(loco_t::texturepack_t* tp, cm_t* cms, const properties_t& mp) {
    auto nr = m_model_list.NewNodeLast();
    auto& node = m_model_list[nr];
    node.cm = *cms;
    //node.groups.
    //cms->
    /*m_model_list[node].
    model_id_t model_id = (model_id_t)cms;
    m_model_list[model_id] = cms;*/

    for (auto& i : cms->instances) {
      std::visit([&](auto&& o) {
        if constexpr (std::is_same_v<std::remove_reference_t<decltype(o)>, model_loader_t::sprite_t>) {
          loco_t::sprite_t::properties_t p;
          p.camera = mp.camera;
          p.viewport = mp.viewport;
          p.position = o.position;
          p.size = o.size;
          loco_t::texturepack_t::ti_t ti;
          if (ti.qti(tp, o.texturepack_name)) {
            fan::throw_error("invalid textureapack name", o.texturepack_name);
          }
          p.load_tp(&ti);
          push_shape(nr, o.group_id, p, o);
        }
        }, i.second.type);
    }

    return nr;
  }
 /* void erase(model_id_t id, uint32_t group_id) {
    auto& cids = m_model_list[id]->groups[group_id].cids;
    for (auto& i : cids) {
      loco_var.erase_shape(i.cid.get());
    }
    m_model_list[id]->groups.erase(group_id);
  }

  void erase(model_id_t id) {
    auto& groups = m_model_list[id]->groups;
    for (auto it = groups.begin(); it != groups.end(); ) {
      auto& cids = it->second.cids;
      for (auto& i : cids) {
        loco_var.erase_shape(i.cid.get());
      }
      it = groups.erase(it);
    }
  }*/

  template <typename T>
  loco_t::cid_t* push_shape(model_id_t model_id, uint32_t group_id, const T& properties, const auto& internal_properties) {
    auto& cids = m_model_list[model_id].groups[group_id].cids;
    typename std::remove_reference_t<decltype(cids)>::value_type p;
    p.internal_ = (typename T::type_t*)0;
    p.cid = std::make_shared<loco_t::cid_t>();
    p.position = properties.position;
    p.size = properties.size;
    p.angle = properties.angle;
    p.id = internal_properties.id;
    cids.emplace_back(p);
    loco_var.push_shape(cids.back().cid.get(), properties);
    return cids.back().cid.get();
  }

  void iterate(model_id_t model_id, auto lambda) {
    for (auto& it2 : m_model_list[model_id].groups) {
      for (auto& i : it2.second.cids) {
        std::visit([&](auto&& o) {
          lambda.template operator() < std::remove_reference_t<decltype(o)>>(it2.first, i.cid, i);
          }, i.internal_);
      }
    }
  }

  void iterate(model_id_t model_id, uint32_t group_id, auto lambda) {
    for (auto& i : m_model_list[model_id].cm.instances) {
      std::visit([&](auto&& o) {
        lambda.template operator()<std::remove_reference_t<decltype(o)>>(i.first, o);
        }, i.second.type);
    }
  }

  void iterate_cids(model_id_t model_id, auto lambda, fan::function_t<void(model_id_data_t::model_t&)> group_lambda = [](model_id_data_t::model_t&){}) {
    for (auto& it : m_model_list[model_id].groups) {
      iterate_cids(model_id, it.first, lambda, group_lambda);
    }
  }

  void iterate_cids(model_id_t model_id, uint32_t group_id, auto lambda, fan::function_t<void(model_id_data_t::model_t&)> group_lambda = [](model_id_data_t::model_t&) {}) {
    auto& group = m_model_list[model_id].groups[group_id];
    for (auto j : m_model_list[model_id].groups[group_id].cids) {
      std::visit([&](auto&& o) {
        using shape_t = std::remove_pointer_t<std::remove_reference_t<decltype(o)>>;
        lambda.template operator() < shape_t > (loco_var.get_shape<shape_t>(), j, group);
      }, j.internal_);
    }
    group_lambda(group);
  }

  void set_position(model_id_t model_id, const fan::vec3& position) {
    iterate_cids(model_id, [&]<typename shape_t>(auto* shape, auto& object, auto& model_info) {
      shape->set(object.cid.get(), &shape_t::vi_t::position, position + object.position);
      shape->set(object.cid.get(), &shape_t::vi_t::rotation_point, object.position);
    });
    m_model_list[model_id].position = position;
  }
  fan::vec3 get_position(model_id_t model_id) {
    return m_model_list[model_id].position;
  }

  //void set_size(model_id_t model_id, const fan::vec3& size) {
  //  iterate_cids(model_id, [&]<typename shape_t>(auto * shape, auto & object, auto & model_info) {
  //    auto offset = size - m_model_list[model_id]->size;
  //    auto current = shape->get(object.cid.get(), &shape_t::vi_t::size);
  //    shape->set(object.cid.get(), &shape_t::vi_t::size, current + offset);
  //  });
  //  m_model_list[model_id]->size = size;
  //}

  void set_angle(model_id_t model_id, f32_t angle) {
    iterate_cids(model_id, 
      // iterate per object in group x
      [&]<typename shape_t>(auto* shape, auto& object, auto& model_info) {
      shape->set(object.cid.get(), &shape_t::vi_t::angle, angle);
    },
    // iterate group
    [&](auto& model_info) {
      model_info.angle = angle;
    }
    );
  }
  void set_angle(model_id_t model_id, uint32_t group_id, f32_t angle) {
    iterate_cids(model_id,
      group_id,
      // iterate per object in group x
      [&]<typename shape_t>(auto * shape, auto & object, auto & model_info) {
      shape->set(object.cid.get(), &shape_t::vi_t::angle, object.angle);
    },
      // iterate group
      [&](auto& model_info) {
      model_info.angle = angle;
    }
    );
  }

  //void set_angle(model_id_t model_id, f32_t angle) {
  //  for (auto& it : m_model_list[model_id]->groups) {
  //    for (auto j : it.second.cids) {
  //      std::visit([&](auto&& o) {
  //        using shape_t = std::remove_pointer_t<std::remove_reference_t<decltype(o)>>;
  //      pile->loco.get_shape<shape_t>()->set(j.cid.get(), &shape_t::vi_t::angle, angle);
  //        }, j.internal_);
  //    }
  //  }
  //}
};

#undef loco_var