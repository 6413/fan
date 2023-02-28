#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

struct pile_t;

#ifndef FAN_INCLUDE_PATH
#define FAN_INCLUDE_PATH C:/libs/fan/include
#endif
#define fan_debug 0
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH, fan/types/types.h)

#define loco_opengl

#define loco_window
#define loco_context

#define loco_no_inline

#define loco_rectangle
#define loco_sprite
#define loco_button
#include _FAN_PATH(graphics/loco.h)

// in stagex.h getting pile from mouse cb
// pile_t* pile = OFFSETLESS(OFFSETLESS(mb.vfi, loco_t, vfi), pile_t, loco);

struct pile_t {
  loco_t loco;

  static constexpr fan::vec2 ortho_x = fan::vec2(-1, 1);
  static constexpr fan::vec2 ortho_y = fan::vec2(-1, 1);

  pile_t() {
    fan::vec2 window_size = loco.get_window()->get_size();
    loco.open_matrices(
      &matrices,
      ortho_x,
      ortho_y
    );
    loco.get_window()->add_resize_callback([&](const fan::window_t::resize_cb_data_t& d) {
      fan::vec2 window_size = d.size;
    // keep aspect ratio
    fan::vec2 ratio = window_size / window_size.max();
    matrices.set_ortho(
      &loco,
      ortho_x * ratio.x,
      ortho_y * ratio.y
    );
    viewport.set(loco.get_context(), 0, window_size, window_size);
      });
    viewport.open(loco.get_context());
    viewport.set(loco.get_context(), 0, window_size, window_size);

    // requires manual open with compiled texture pack name
  }

  loco_t::theme_t theme;
  loco_t::matrices_t matrices;
  fan::graphics::viewport_t viewport;

};

pile_t* pile = new pile_t;

#include _FAN_PATH(graphics/gui/model_maker/loader.h)

#include <variant>

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

int main(int argc, char** argv) {
  if (argc < 2) {
    fan::throw_error("usage: TexturePackCompiled");
  }
  loco_t::texturepack_t tp;
  tp.open_compiled(&pile->loco, argv[1]);
  model_list_t m;

  cm_t cm;
  cm.import_from("model.fmm", &tp);

  auto it = m.push_model(&cm);
  
  uint32_t group_id = 0;

  m.iterate(it, group_id, [](auto shape_id, const auto& properties) {
    using type_t = std::remove_const_t<std::remove_reference_t<decltype(properties)>>;
    if constexpr (std::is_same_v<type_t, model_loader_t::mark_t>) {
      fan::graphics::cid_t cid;
      loco_t::rectangle_t::properties_t rp;
      rp.matrices = &pile->matrices;
      rp.viewport = &pile->viewport;
      rp.position = properties.position;
      rp.size = 0.01;
      rp.color = fan::colors::white;
      pile->loco.rectangle.push_back(&cid, rp);
    }
  });

  pile->loco.loop([&] {

  });

}