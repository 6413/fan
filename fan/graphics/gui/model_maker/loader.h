struct model_list_t {

  struct properties_t {
    loco_t::camera_t camera = gloco->orthographic_camera.camera;
    loco_t::viewport_t viewport = gloco->orthographic_camera.viewport;
    fan::vec3 position = 0;
  };

  struct cm_t {

    void import_from(const fan::string& path, loco_t::texturepack_t* tp) {
      loco_t::texturepack_t::ti_t ti;

      fan::string in;
      fan::io::file::read(path, &in);
      if (in.empty()) {
        return;
      }
      fan::json json_in = nlohmann::json::parse(in);
      shapes = json_in;
    }

    struct shape_t : loco_t::shape_t {
      shape_t(const loco_t::shape_t&s) : loco_t::shape_t(s){}
      shape_t(loco_t::shape_t&& s) : loco_t::shape_t(std::move(s)) {}
      std::string id;
      uint32_t group_id = 0;
      std::string image_name;
    };

    fan::json shapes;
  };

  struct group_data_t {
    loco_t::shape_t shape;
    fan::vec3 position = 0; //offset from root
    fan::vec2 size = 0;
    fan::vec3 angle = 0;
  };

  struct model_data_t {
    cm_t cm;
    //loco_t::shape_t shape;

    fan::vec3 position = 0;
    fan::vec2 size = 1;
    fan::vec3 angle = 0;

    std::vector<std::vector<group_data_t>> groups;
  };

  #include <fan/fan_bll_preset.h>
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

    if (cms->shapes.is_object() == false) {
      return nr;
    }

    fan::vec2 root_pos = -0xfffff;

    auto version = cms->shapes["version"].get<int>();
    if (version != 1) {
      fan::throw_error("invalid file version");
    }
    fan::graphics::shape_deserialize_t iterator;
    loco_t::shape_t shape;
    int i = 0;
    while (iterator.iterate(cms->shapes["shapes"], &shape)) {
      const auto& shape_json = *(iterator.data.it - 1);
      cm_t::shape_t s = shape;
      s.set_camera(mp.camera);
      s.set_viewport(mp.viewport);
      s.id = shape_json["id"].get<fan::string>();
      s.group_id = shape_json["group_id"].get<uint32_t>();
      auto st = shape.get_shape_type();
      if (st == loco_t::shape_type_t::sprite ||
        st == loco_t::shape_type_t::unlit_sprite || st == loco_t::shape_type_t::light) {
        if (st == loco_t::shape_type_t::sprite ||
          st == loco_t::shape_type_t::unlit_sprite) {
          s.image_name = shape_json["image_name"].get<fan::string>();
          if (s.image_name.size()) {
            loco_t::texturepack_t::ti_t ti;
            if (tp->qti(s.image_name, &ti)) {
              fan::throw_error("failed to read images");
            }
            s.load_tp(&ti);
          }
          else {
            fan::print_warning("empty image");
          }
        }

        if (s.group_id == 0 && root_pos == -0xfffff) {// set root pos
          root_pos = s.get_position();
          s.set_position(fan::vec3(fan::vec2(0), mp.position.z));
        }
        else {
          s.set_position(fan::vec3(fan::vec2(fan::vec2(s.get_position()) - root_pos), mp.position.z));
        }
        fan::print(s.get_color());
        push_shape(nr, s.group_id, std::move(s));
      }
    }
    set_position(nr, mp.position);
    return nr;
  }

  void push_shape(model_id_t model_id, uint32_t group_id, const cm_t::shape_t& shape) {
    auto& model = model_list[model_id];
    if (model.groups.size() < group_id + 1) {
      model.groups.resize(group_id + 1);
    }
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
    auto& m = model_list[model_id];
    if (group_id >= m.groups.size()) {
      return;
    }
    for (auto& i : m.groups[group_id]) {
      lambda(i);
    }
  }
  void iterate_marks(model_id_t model_id, uint32_t group_id, auto lambda) {
    for (auto& i : model_list[model_id].groups[group_id]) {
      if (i.shape.get_shape_type() == loco_t::shape_type_t::rectangle) {
        lambda(i);
      }
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
  void set_position(model_id_t model_id, uint32_t group_id, const fan::vec3& position) {
    auto& model = model_list[model_id];
    auto& group = model.groups[group_id];
    for (auto& j : group) {
      j.position = position;
      j.shape.set_position(position);
    }
  }
  void set_position(model_id_t model_id, uint32_t group_id, const fan::vec2& position) {
    auto& model = model_list[model_id];
    auto& group = model.groups[group_id];
    for (auto& j : group) {
      j.position = position;
      j.shape.set_position(position);
    }
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

  // changes all sizes to size -- only benefitable with one object
  void set_size(model_id_t model_id, uint32_t group_id, const fan::vec2& size) {
    auto& model = model_list[model_id];
    // skip root
    auto& group = model.groups[group_id];
    for (auto& j : group) {
      j.shape.set_size(size);
    }
  }

  fan::vec3 get_angle(model_id_t model_id, uint32_t group_id) {
    auto& model = model_list[model_id];
    // skip root
    auto& group = model.groups[group_id];
    return group[0].shape.get_angle();
  }
  void set_angle(model_id_t model_id, const fan::vec3& angle) {
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
  void set_angle(model_id_t model_id, uint32_t group_id, const fan::vec3& angle) {
    auto& model = model_list[model_id];
    // skip root
    if (group_id >= model.groups.size()) {
      return;
    }
    auto& group = model.groups[group_id];
    for (auto& j : group) {
      j.shape.set_angle(angle + j.angle);
      //j.shape.set_rotation_point(0);
    }
  }
  void set_rotation_point(model_id_t model_id, uint32_t group_id, const fan::vec2& rp) {
    auto& model = model_list[model_id];
    auto& group = model.groups[group_id];
    for (auto& j : group) {
      j.shape.set_rotation_point(rp);
    }
  }
  auto& get_instance(model_id_t model_id) {
    return model_list[model_id];
  }

  void iterate(model_id_t model_id, auto l) {

    auto& node = model_list[model_id];
    fan::graphics::shape_deserialize_t iterator;
    loco_t::shape_t shape;
    int i = 0;
    while (iterator.iterate(node.cm.shapes["shapes"], &shape)) {
      const auto& shape_json = *(iterator.data.it - 1);
      l(shape_json);
    }
  }
};

#undef loco_var