struct model_list_t {

  struct properties_t {
    loco_t::camera_t camera = gloco->orthographic_render_view.camera;
    fan::graphics::viewport_t viewport = gloco->orthographic_render_view.viewport;
    fan::vec3 position = 0;
  };

  struct cm_t {

    void import_from(const std::string& path, loco_t::texturepack_t* tp, const std::source_location& callers_path = std::source_location::current()) {
      loco_t::texturepack_t::ti_t ti;

      std::string in;
      fan::io::file::read(fan::io::file::find_relative_path(path, callers_path), &in);
      if (in.empty()) {
        return;
      }
      fan::json json_in = fan::json::parse(in);
      shapes = json_in;
    }

    struct shape_t : fan::graphics::shape_t {
      shape_t(const fan::graphics::shape_t&s) : fan::graphics::shape_t(s){}
      shape_t(fan::graphics::shape_t&& s) : fan::graphics::shape_t(std::move(s)) {}
      std::string id;
      uint32_t group_id = 0;
      std::string image_name;
    };

    fan::json shapes;
  };

  struct group_data_t{
    cm_t::shape_t shape;

    fan::vec3 position = 0; //offset from root
    fan::vec2 size = 0;
    fan::vec3 angle = 0;
    std::string id;
  };

  struct model_data_t {
    cm_t cm;
    //fan::graphics::shape_t shape;

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
  #define BLL_set_CPP_CopyAtPointerChange 1
  #include <BLL/BLL.h>

  using model_id_t = internal_model_list_NodeReference_t;
  internal_model_list_t model_list;

  std::unordered_map<std::string, fan::vec3> mark_positions;

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
    fan::graphics::shape_t shape;
    int i = 0;
    while (iterator.iterate(cms->shapes["shapes"], &shape)) {
      const auto& shape_json = *(iterator.data.it - 1);
      cm_t::shape_t s = shape;
      s.set_camera(mp.camera);
      s.set_viewport(mp.viewport);
      s.id = shape_json["id"].get<std::string>();
      s.group_id = shape_json["group_id"].get<uint32_t>();
      auto st = shape.get_shape_type();
      if (st == loco_t::shape_type_t::sprite ||
        st == loco_t::shape_type_t::unlit_sprite || st == loco_t::shape_type_t::light) {
        if (st == loco_t::shape_type_t::sprite ||
          st == loco_t::shape_type_t::unlit_sprite) {
          s.image_name = shape_json["image_path"].get<std::string>();
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
        push_shape(nr, s.group_id, std::move(s));
      }
      else if (st == loco_t::shape_type_t::rectangle) {// mark - for example invisble position to get a reference point
        if (s.id.size()) {
          auto [it, emplaced] = mark_positions.try_emplace(s.id);
          if (!emplaced) {
            fan::print("error: duplicate model id - undefined behaviour");
          }
          else {
            it->second = s.get_position() - node.position;
          }
        }
      }
    }
    set_position(nr, mp.position);
    return nr;
  }
  void push_shape(model_id_t model_id, uint32_t group_id, cm_t::shape_t&& shape) {
    auto& model = model_list[model_id];
    if (model.groups.size() < group_id + 1) {
      model.groups.resize(group_id + 1);
    }
    auto& group = model.groups[group_id];
    group.push_back(group_data_t{ 
      .shape = std::move(shape)
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
    auto& model = model_list[model_id];
    model.groups.erase(model_list[model_id].groups.begin() + group_id);
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

  fan::vec3 get_position(model_id_t model_id, const std::string& id) {
    if (auto found = mark_positions.find(id); found != mark_positions.end()) {
      return found->second;
    }

    auto& model = model_list[model_id];
    for (auto& group : model.groups) {
      for (auto& shape : group) {
        if (shape.id == id) {
          return shape.shape.get_position();
        }
      }
    }
    fan::throw_error("id not found from model");
    return fan::vec3();
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
    auto& model = model_list[model_id];
    for (auto& group : model.groups) {
      for (auto& j : group) {
        l(j);
      }
    }
  }
};

struct _model_list_filler_t {
  struct _model_list_t : model_list_t {
    struct cm_t : model_list_t::cm_t {
      inline static std::string model_path = "models/";
      cm_t(const std::string& Name) {
        import_from(model_path + Name, &engine::tp);
      }
      ~cm_t() {
        //_model_list_filler_t::model_list.erase(internal_id);
      }
    };
    struct id_t {
      model_list_t::model_id_t internal_id;
      ~id_t() {
        _model_list_filler_t::model_list.erase(internal_id);
      }
      operator model_list_t::model_id_t() {
        return internal_id;
      }
      void add(cm_t* cm, const model_list_t::properties_t& p) {
        internal_id = _model_list_filler_t::model_list.push_model(&engine::tp, cm, p);
      }
      void add_shape(uint32_t group_id, const auto& shape) {
        _model_list_filler_t::model_list.push_shape(internal_id, group_id, model_list_t::cm_t::shape_t(shape));
      }
      void iterate(uint32_t group_id, auto lambda) {
        _model_list_filler_t::model_list.iterate(internal_id, group_id, lambda);
      }
      void iterate(auto lambda) {
        _model_list_filler_t::model_list.iterate(internal_id, lambda);
      }
      void iterate_marks(uint32_t group_id, auto lambda) {
        _model_list_filler_t::model_list.iterate_marks(internal_id, group_id, lambda);
      }
      void rem() {
        _model_list_filler_t::model_list.erase(internal_id);
      }
      void rem(uint32_t group_id) {
        _model_list_filler_t::model_list.erase(internal_id, group_id);
      }
      fan::vec3 get_position() {
        return _model_list_filler_t::model_list.get_instance(internal_id).position;
      }
      fan::vec3 get_position(const std::string& id) {
        return _model_list_filler_t::model_list.get_position(internal_id, id);
      }
      void set_position(const fan::vec3& Position) {
        _model_list_filler_t::model_list.set_position(internal_id, Position);
      }
      void set_position(const fan::vec2& Position) {
        _model_list_filler_t::model_list.set_position(internal_id, Position);
      }
      void set_position(uint32_t group_id, const fan::vec2& Position) {
        _model_list_filler_t::model_list.set_position(internal_id, group_id, Position);
      }
      void set_position(uint32_t group_id, const fan::vec3& Position) {
        _model_list_filler_t::model_list.set_position(internal_id, group_id, Position);
      }
      void set_size(const fan::vec2& Size) {
        _model_list_filler_t::model_list.set_size(internal_id, Size);
      }
      void set_size(uint32_t group_id, const fan::vec2& Size) {
        _model_list_filler_t::model_list.set_size(internal_id, group_id, Size);
      }
      auto& get_instance() {
        return _model_list_filler_t::model_list.get_instance(internal_id);
      }
      void set_angle(const fan::vec3& angle) {
        _model_list_filler_t::model_list.set_angle(internal_id, angle);
      }
      fan::vec3 get_angle(uint32_t group_id) {
        return _model_list_filler_t::model_list.get_angle(internal_id, group_id);
      }
      void set_angle(uint32_t group_id, const fan::vec3& angle) {
        _model_list_filler_t::model_list.set_angle(internal_id, group_id, angle);
      }
      void set_rotation_point(uint32_t group_id, const fan::vec2& rotation_point) {
        _model_list_filler_t::model_list.set_rotation_point(internal_id, group_id, rotation_point);
      }
    };
  };
  inline static _model_list_t model_list;
};