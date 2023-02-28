#include <variant>

struct model_loader_t {

  struct private_ {
    #define fgm_build_model_maker
    #include _FAN_PATH(graphics/gui/fgm/common.h)
    #undef fgm_build_model_maker
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

    struct iterator_t {
      fan_masterpiece_make(
        (private_::stage_maker_shape_format::shape_sprite_t)sprite,
        (private_::stage_maker_shape_format::shape_mark_t)mark
      );
    }iterator;

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
  std::vector<std::unordered_map<std::string,
    std::variant<
    model_loader_t::mark_t,
    model_loader_t::sprite_t
    >
    >> models;

  void import_from(const char* path, loco_t::texturepack_t* tp) {
    model_loader_t loader;
    loco_t::texturepack_t::ti_t ti;
    loader.load(tp, path, [&](const auto& data) {
      models.resize(fan::max(models.size(), data.group_id + 1));
    models[data.group_id][data.id] = data;
      });
  }
};

struct model_list_t {
private:
  #define BLL_set_CPP_ConstructDestruct
  #define BLL_set_CPP_Node_ConstructDestruct
  #define BLL_set_BaseLibrary 1
  #define BLL_set_prefix model_list_internal
  #define BLL_set_type_node uint8_t
  #define BLL_set_NodeData cm_t* cms;
  #define BLL_set_Link 1
  #define BLL_set_AreWeInsideStruct 1
  #include _FAN_PATH(BLL/BLL.h)
public:

  model_list_internal_t model_list;

  using model_id_t = model_list_internal_NodeReference_t;

  model_id_t push_model(cm_t* cms) {
    auto it = model_list.GetNodeLast();
    model_list[it].cms = cms;
    return it;
  }
  constexpr void iterate(model_id_t model_id, uint32_t group_id, auto lambda) {
    for (auto& i : model_list[model_id].cms->models[group_id]) {
      std::visit([&](auto&& o) {
        lambda(i.first, o);
        }, i.second);
    }
  }
};