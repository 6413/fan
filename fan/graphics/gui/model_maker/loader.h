struct model_loader_t {
  #define model_maker_loader
  #include _FAN_PATH(graphics/gui/fgm/common.h)

  void load(loco_t::texturepack_t* tp, const fan::string& filename, auto lambda) {
    #define model_maker_loader
    #include _FAN_PATH(graphics/gui/stage_maker/loader_versions/1.h)
  }
};

struct model_list_t {

  using current_version_t = model_loader_t::current_version_t;

  using shapes_t = current_version_t::shapes_t;

  struct properties_t {
    loco_t::camera_t* camera = &gloco->default_camera->camera;
    fan::graphics::viewport_t* viewport = &gloco->default_camera->viewport;
    fan::vec3 position = 0;
  };

  struct cm_t {

    void import_from(const fan::string& path, loco_t::texturepack_t* tp) {
      model_loader_t loader;
      loco_t::texturepack_t::ti_t ti;
      loader.load(tp, path, [&]<typename T>(const T& data) {
        shapes.resize(shapes.size() + 1);
        shapes.back() = data;
      });
    }

    std::vector<fan::union_mp<shapes_t>> shapes;
  };

  struct group_data_t {
    loco_t::shape_t shape;
    fan::vec3 position = 0; //offset from root
    fan::vec2 size = 0;
    f32_t angle = 0;
  };

  struct model_data_t {
    cm_t cm;
    //loco_t::shape_t shape;

    fan::vec3 position = 0;
    fan::vec2 size = 1;
    f32_t angle = 0;

    std::vector<std::vector<group_data_t>> groups;
  };

  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #include _FAN_PATH(fan_bll_preset.h)
  #define BLL_set_prefix internal_model_list
  #define BLL_set_type_node uint32_t
  #define BLL_set_NodeDataType model_data_t
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(BLL/BLL.h)

  using model_id_t = internal_model_list_NodeReference_t;
  internal_model_list_t model_list;

  model_id_t push_model(loco_t::texturepack_t* tp, cm_t* cms, const properties_t& mp) {
    auto nr = model_list.NewNodeLast();
    auto& node = model_list[nr];
    node.cm = *cms;
    fan::vec2 root_pos = 0;
    for (auto& i : cms->shapes) {
      std::visit([&]<typename T>(T & v) {
        if constexpr (
          std::is_same_v<T, current_version_t::sprite_t> ||
          std::is_same_v<T, current_version_t::unlit_sprite_t>) {
          auto&& shape = v.get_shape(tp);
          shape.set_camera(mp.camera);
          shape.set_viewport(mp.viewport);
          if (v.group_id == 0) {// set root pos
            root_pos = shape.get_position();
            shape.set_position(fan::vec2(0));
          }
          else {
            shape.set_position(fan::vec2(fan::vec2(shape.get_position()) - root_pos));
          }
          push_shape(nr, v.group_id, std::move(shape));
        }
      }, i);
    }
    set_position(nr, mp.position);
    return nr;
  }

  void push_shape(model_id_t model_id, uint32_t group_id, const auto& shape) {
    auto& model = model_list[model_id];
    model.groups.resize(group_id + 1);
    auto& group = model.groups[group_id];
    group.push_back(group_data_t{ 
      .shape = shape
    });
    group.back().position = group.back().shape.get_position() - model.position;
    group.back().size = group.back().shape.get_size(); // ?
    group.back().angle = group.back().shape.get_angle(); // ?
  }

  void erase(model_id_t model_id) {
    model_list.unlrec(model_id);
  }
  // what happens when group ids change at erase
  void erase(model_id_t model_id, uint32_t group_id) {
    model_list[model_id].groups.erase(model_list[model_id].groups.begin() + group_id);
  }

  void iterate(model_id_t model_id, uint32_t group_id, auto lambda) {
    for (auto& i : model_list[model_id].cm.shapes) {
      std::visit([&]<typename T>(T & v) {
        if (v.group_id != group_id) {
          return;
        }
        lambda(v);
      }, i);
    }
  }
  void iterate_marks(model_id_t model_id, uint32_t group_id, auto lambda) {
    for (auto& i : model_list[model_id].cm.shapes) {
      std::visit([&]<typename T>(T & v) {
        if (v.group_id != group_id) {
          return;
        }
        if constexpr (std::is_same_v<T, current_version_t::mark_t>) {
          lambda(v);
        }
      }, i);
    }
  }

  void set_position(model_id_t model_id, const fan::vec3& position) {
    auto& model = model_list[model_id];
    for (auto& group : model.groups) {
      for (auto& j : group) {
        j.shape.set_position(fan::vec3(position + fan::vec3(fan::vec2(j.position) * model.size, j.position.z)));
      }
    }
    model.position = position;
  }
  void set_position(model_id_t model_id, const fan::vec2& position) {
    auto& model = model_list[model_id];
    for (auto& group : model.groups) {
      for (auto& j : group) {
        j.shape.set_position(fan::vec2(position + fan::vec2(j.position) * model.size));
      }
    }
    *(fan::vec2*)&model.position = position;
  }
  void set_size(model_id_t model_id, const fan::vec2& size) {
    auto& model = model_list[model_id];
    for (auto& group : model.groups) {
      for (auto& j : group) {
        j.shape.set_position(fan::vec2(fan::vec2(model.position) + fan::vec2(j.position) * size));
        j.shape.set_size(size * j.size);
      }
    }
    model.size = size;
  }
  void set_angle(model_id_t model_id, f32_t angle) {
    auto& model = model_list[model_id];
    model.angle = angle;
    // skip root
    for (int i = 0; i < model.groups.size(); ++i) {
      auto& group = model.groups[i];
      for (auto& j : group) {
        j.shape.set_angle(angle + j.angle);
        j.shape.set_rotation_point(j.position);
      }
    }
  }
  void set_angle(model_id_t model_id, uint32_t group_id, f32_t angle) {
    auto& model = model_list[model_id];
    // skip root
    auto& group = model.groups[group_id];
    for (auto& j : group) {
      j.shape.set_angle(angle + j.angle);
      j.shape.set_rotation_point(0);
    }
  }
  auto& get_instance(model_id_t model_id) {
    return model_list[model_id];
  }
};

#undef loco_var