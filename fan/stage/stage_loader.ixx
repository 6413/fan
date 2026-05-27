module;

#include <fan/utility.h>
#include <cstdint> //bll

export module fan.stage.loader;

import std;

import fan.window;
import fan.graphics.shapes;
import fan.graphics;
import fan.graphics.loco;
import fan.event.types;
import fan.print.error;
import fan.memory;
import fan.graphics.common_context;
import fan.physics.b2_integration;

#include <fan/graphics/gui/fgm/common.h>

export namespace fan {
  struct stage_loader_t;

  inline struct gstage_t {
    stage_loader_t* loader;

    operator stage_loader_t*() {
      return loader;
    }
    gstage_t& operator=(stage_loader_t* l) {
      loader = l;
      return *this;
    }
    stage_loader_t* operator->() {
      return loader;
    }
  } gstage;

  struct stage_loader_t {
    #define BLL_set_Link 1
    #define BLL_set_declare_NodeReference 1
    #define BLL_set_declare_rest 0
    #include <fan/fan_bll_preset.h>
    #define BLL_set_AreWeInsideStruct 1
    #define BLL_set_prefix stage_list
    #define BLL_set_type_node std::uint16_t
    #define bcontainer_set_StoreFormat 1
    #define BLL_set_NodeData \
    fan::graphics::update_callback_nr_t update_nr; \
    fan::window_t::resize_callback_NodeReference_t resize_id; \
    void *stage;
    #include <BLL/BLL.h>

    #define bcontainer_set_StoreFormat 1
    #include <fan/fan_bll_preset.h>
    #define BLL_set_prefix cid_list
    #define BLL_set_type_node std::uint32_t
    #define BLL_set_NodeDataType fan::graphics::shape_t
    #define BLL_set_Link 1
    #define BLL_set_AreWeInsideStruct 1
    #include <BLL/BLL.h>

    #define BLL_set_declare_NodeReference 0
    #define BLL_set_declare_rest 1
    #include <fan/fan_bll_preset.h>
    #define BLL_set_prefix stage_list
    #define BLL_set_type_node std::uint16_t
    #define BLL_set_NodeData \
    fan::graphics::update_callback_nr_t update_nr; \
    fan::window_t::resize_callback_NodeReference_t resize_id; \
    void *stage;
    #define BLL_set_Link 1
    #define BLL_set_AreWeInsideStruct 1
    #include <BLL/BLL.h>

    using cid_nr_t = cid_list_NodeReference_t;

    struct nr_t : stage_list_NodeReference_t {
      using base_nr_t = stage_list_NodeReference_t;
      using base_nr_t::base_nr_t;

      nr_t() = default;
      nr_t(base_nr_t nr) : base_nr_t(nr) {}

      void erase() {
        #if FAN_DEBUG >= 2
        if (iic()) {
          fan::throw_error("double erase or uninitialized erase");
        }
        #endif
        gstage->close_stage(*this);
        sic();
      }
    };

    struct stage_open_properties_t {
      fan::graphics::camera_t* camera = &gloco()->orthographic_render_view.camera;
      fan::graphics::viewport_t* viewport = &gloco()->orthographic_render_view.viewport;

      stage_loader_t::nr_t parent_id;
      std::uint32_t itToDepthMultiplier = 0x100;

      void* sod;
    };

    struct stage_common_t {
      using open_t = void(*)(void*, void*);
      using close_t = void(*)(void*);
      using window_resize_t = void(*)(void*);
      using update_t = void(*)(void*);

      open_t open;
      close_t close;
      window_resize_t window_resize;
      update_t update;
      stage_list_NodeReference_t stage_id;
      std::uint32_t it;
      cid_list_t cid_list;
      stage_loader_t::stage_list_NodeReference_t parent_id;
      fan::graphics::update_callback_nr_t update_nr;
    };

    using key_t = std::pair<void*, std::string>;

    struct pair_hasher_t {
      std::size_t operator()(const key_t& pair) const {
        return std::hash<decltype(pair.first)>()(pair.first) ^ std::hash<std::string>()(pair.second);
      }
    };

    struct pair_equal_t {
      bool operator()(const key_t& lhs, const key_t& rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
      }
    };

    using cid_map_t = std::unordered_map<key_t, cid_nr_t, pair_hasher_t, pair_equal_t>;

    stage_loader_t() {
      gstage = this;
    }

    template <typename T>
    static consteval std::string_view get_type_name() {
      #if defined(__clang__)
      constexpr std::string_view p = "[T = ";
      constexpr std::string_view s = "]";
      constexpr std::string_view n = __PRETTY_FUNCTION__;
      #elif defined(__GNUC__)
      constexpr std::string_view p = "[with T = ";
      constexpr std::string_view s = "]";
      constexpr std::string_view n = __PRETTY_FUNCTION__;
      #elif defined(_MSC_VER)
      constexpr std::string_view p = "get_type_name<struct ";
      constexpr std::string_view s = ">(void)";
      constexpr std::string_view n = __FUNCSIG__;
      #else
      return "";
      #endif
      return n.substr(n.find(p) + p.size(), n.rfind(s) - (n.find(p) + p.size()));
    }

    template <typename T>
    stage_list_NodeReference_t open_stage(const stage_open_properties_t& op) {
      if (!T::engine)  T::engine = gloco();
      if (!T::window)  T::window = &gloco()->window;
      if (!T::physics) T::physics = &gloco()->get_physics_context();

      previous_stage_name = current_stage_name;
      if constexpr (requires { T::stage_name; }) {
        current_stage_name = T::stage_name;
      } else {
        current_stage_name = get_type_name<T>();
      }

      T* stage = new T{op};
      stage->stage_common.open(stage, op.sod);
      return stage->stage_common.stage_id;
    }

    template <typename T>
    stage_list_NodeReference_t open_stage() {
      return open_stage<T>(stage_open_properties_t());
    }

    template <typename T>
    T& get_stage_data(nr_t nr) {
      return *static_cast<T*>(stage_list[nr].stage);
    }

    fan::graphics::shape_t& get_id(auto* stage_ptr, const std::string id) {
      auto found = cid_map.find(std::make_pair(stage_ptr, id));
      if (found == cid_map.end()) {
        fan::throw_error("invalid fetch for id - usage shape_{id}", id);
      }

      return stage_ptr->stage_common.cid_list[found->second];
    }

    void load_fgm(auto* stage, const stage_open_properties_t& op, const char* stage_name) {
      std::string filename = std::string("stages_runtime/") + stage_name + ".fgm";
      #define only_struct_data
      #include <fan/graphics/gui/stage_maker/loader_versions/1.h>
    }

    void close_stage(nr_t id) {
      auto* sc = (stage_common_t*)stage_list[id].stage;
      gloco()->m_update_callback.unlrec(stage_list[id].update_nr);
      sc->close(stage_list[id].stage);
      stage_list.unlrec(id);
    }

    template <typename T>
    void restart_stage(nr_t& id) {
      if (id) {
        close_stage(id);
      }
      id = open_stage<T>();
    }

    template <typename NextStageT>
    fan::event::task_t change_stage(
      nr_t old_stage_id,
      nr_t& out_new_stage_id,
      f32_t duration = 1.f,
      fan::color color = fan::colors::black
    ) {
      fan::graphics::rectangle_t overlay {{
        .position = fan::vec3(0, 0, 0xfffe),
        .size = fan::vec2(99999),
        .color = color.set_alpha(0),
        .blending = true,
      }};

      f32_t half = duration / 2.f;
      for (f32_t t = 0.f; t < half; t += gloco()->get_delta_time()) {
        overlay.set_color(color.set_alpha(t / half));
        co_await fan::graphics::co_next_frame();
      }

      close_stage(old_stage_id);
      out_new_stage_id = open_stage<NextStageT>();

      for (f32_t t = 0.f; t < half; t += gloco()->get_delta_time()) {
        overlay.set_color(color.set_alpha(1.f - t / half));
        co_await fan::graphics::co_next_frame();
      }
    }

    stage_list_t stage_list;
    cid_map_t cid_map;
    std::string current_stage_name;
    std::string previous_stage_name;
  };

  template <typename Derived>
  struct stage_t {
    using self_t = Derived;

    stage_t(const stage_loader_t::stage_open_properties_t& op = {}) {
      stage_common.open = [](void* ptr, void* sod) {
        if constexpr (requires { static_cast<Derived*>(ptr)->open(sod); }) {
          static_cast<Derived*>(ptr)->open(sod);
        }
      };
      stage_common.close = [](void* ptr) {
        if constexpr (requires { static_cast<Derived*>(ptr)->close(); }) {
          static_cast<Derived*>(ptr)->close();
        }
        delete static_cast<Derived*>(ptr);
      };
      stage_common.update = [](void* ptr) {
        if constexpr (requires { static_cast<Derived*>(ptr)->update(); }) {
          static_cast<Derived*>(ptr)->update();
        }
      };

      auto outside = static_cast<Derived*>(this);
      auto nr = gstage->stage_list.NewNodeLast();
      stage_common.stage_id = nr;
      stage_common.parent_id = op.parent_id;
      gstage->stage_list[nr].stage = outside;

      if (stage_common.stage_id.Prev(&gstage->stage_list) != gstage->stage_list.src) {
        stage_common.it = static_cast<stage_loader_t::stage_common_t*>(
          gstage->stage_list[stage_common.stage_id.Prev(&gstage->stage_list)].stage)->it + 1;
      } else {
        stage_common.it = 0;
      }

      stage_common.update_nr = gloco()->m_update_callback.NewNodeFirst();
      gstage->stage_list[stage_common.stage_id].update_nr = stage_common.update_nr;
      gloco()->m_update_callback[stage_common.update_nr] = [outside](void*) {
        outside->stage_common.update(outside);
      };

      auto resize_id = gloco()->window.add_resize_callback([outside](const auto& d) {
        std::printf("todo -- stage common window resize\n");
      });
      gstage->stage_list[stage_common.stage_id].resize_id = resize_id;
    }

    static constexpr std::string_view get_stage_name() {
      if constexpr (requires { Derived::stage_name; }) {
        return Derived::stage_name;
      } else {
        return stage_loader_t::get_type_name<Derived>();
      }
    }

    inline static loco_t* engine = nullptr;
    inline static fan::window_t* window = nullptr;
    inline static fan::physics::context_t* physics = nullptr;
    stage_loader_t::stage_common_t stage_common;
  };
}