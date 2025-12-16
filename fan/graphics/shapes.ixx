module;

#include <fan/utility.h>

#include <source_location>
#include <cstdlib>
#include <utility>
#include <vector>
#include <functional>
#include <memory>
#include <cstring>

#include <fan/graphics/opengl/init.h>

export module fan.graphics.shapes;

export import fan.graphics.shapes.types;

import fan.graphics.culling;

import fan.graphics.opengl.core;
import fan.types.vector;
import fan.texture_pack.tp0;
import fan.graphics.common_context;
import fan.io.file;
import fan.window;
import fan.time;
import fan.utility;
import fan.physics.collision.rectangle;
import fan.physics.collision.circle;

import fan.types.fstring;

#if defined(fan_json)
  import fan.types.json;
#endif

#if defined(fan_opengl)
  import fan.graphics.opengl.core;
#endif

#if defined(fan_vulkan)
  import fan.graphics.vulkan.core;
#endif

#if defined(fan_physics)
  import fan.physics.types; // aabb
#endif

#define shaper_get_key_safe(return_type, kps_type, variable) \
[key_pack] ()-> auto& { \
	auto o = g_shapes->shaper.GetKeyOffset( \
		offsetof(fan::graphics::kps_t::CONCAT(_, kps_type), variable), \
		offsetof(fan::graphics::kps_t::kps_type, variable) \
	);\
	static_assert(std::is_same_v<decltype(fan::graphics::kps_t::kps_type::variable), fan::graphics::return_type>, "possibly unwanted behaviour"); \
	return *(fan::graphics::return_type*)&key_pack[o];\
}()

#define shape_get_vi(shape) (*(fan::graphics::shapes::shape##_t::vi_t*)GetRenderData(fan::graphics::g_shapes->shaper))
#define shape_get_ri(shape) (*(fan::graphics::shapes::shape##_t::ri_t*)GetData(fan::graphics::g_shapes->shaper))

export namespace fan::graphics {

  struct shapes;
  inline shapes* g_shapes = nullptr;

#if defined(fan_json)
  fan::json image_to_json(const fan::graphics::image_t& image);
  fan::graphics::image_t json_to_image(const fan::json& image_json, const std::source_location& callers_path = std::source_location::current());
#endif
  // things that shapes require, should be moved in future to own.ixx

  //-----------------------sprite sheet animations-----------------------

  struct sprite_sheet_animation_t {
    struct image_t {
      fan::graphics::image_t image = fan::graphics::ctx().default_texture;
      int hframes = 1, vframes = 1;
    #if defined(fan_json)
      operator fan::json() const;
      sprite_sheet_animation_t::image_t& assign(const fan::json& j, const std::source_location& callers_path = std::source_location::current());
    #endif
    };
    sprite_sheet_animation_t();
    ~sprite_sheet_animation_t();
    std::vector<int> selected_frames;
    std::vector<sprite_sheet_animation_t::image_t> images;
    std::string name;
    int fps = 15;
    bool loop = true;
  };

  struct animation_nr_t {
    animation_nr_t();
    animation_nr_t(uint32_t id);
    operator uint32_t() const;
    operator bool() const;
    animation_nr_t operator++(int);
    bool operator==(const animation_nr_t& other) const;
    bool operator!=(const animation_nr_t& other) const;
    uint32_t id = -1;
  };

  using animation_shape_nr_t = animation_nr_t;

  struct animation_nr_hash_t {
    size_t operator()(const animation_nr_t& anim_nr) const noexcept;
  };

  struct animation_pair_hash_t {
    std::size_t operator()(const std::pair<animation_nr_t, std::string>& p) const noexcept;
  };
#pragma pack(push, 1)
  struct sprite_sheet_data_t {
    // current_frame in 'selected_frames'
    int current_frame;
    fan::time::timer update_timer;
    // sprite sheet update function nr
    fan::graphics::update_callback_nr_t frame_update_nr;
  };
#pragma pack(pop)
  extern std::unordered_map<animation_nr_t, sprite_sheet_animation_t, animation_nr_hash_t> all_animations;
  extern animation_nr_t all_animations_counter;
  extern std::unordered_map<std::pair<animation_shape_nr_t, std::string>, animation_nr_t, animation_pair_hash_t> shape_animation_lookup_table;
  extern std::unordered_map<animation_shape_nr_t, std::vector<animation_nr_t>, animation_nr_hash_t> shape_animations;
  extern animation_nr_t shape_animation_counter;

  sprite_sheet_animation_t& get_sprite_sheet_animation(animation_nr_t nr);
  sprite_sheet_animation_t& get_sprite_sheet_animation(animation_nr_t shape_animation_id, const std::string& anim_name);
  std::vector<fan::graphics::animation_nr_t>& get_sprite_sheet_shape_animation(animation_nr_t shape_animation_id);
  void rename_sprite_sheet_shape_animation(animation_nr_t shape_animation_id, const std::string& old_name, const std::string& new_name);
  // adds animation to shape collection
  animation_nr_t add_sprite_sheet_shape_animation(animation_nr_t new_anim);
  animation_nr_t add_existing_sprite_sheet_shape_animation(animation_nr_t existing_anim, animation_nr_t shape_animation_id, const sprite_sheet_animation_t& new_anim);
  // returns unique key to access list of animation keys
  animation_nr_t add_sprite_sheet_shape_animation(animation_nr_t shape_animation_id, const sprite_sheet_animation_t& new_anim);
  bool is_animation_finished(animation_nr_t nr, const fan::graphics::sprite_sheet_data_t& sd);

#if defined(fan_json)
  fan::json sprite_sheet_serialize();
  void parse_animations(const std::string& json_path, fan::json& json, const std::source_location& callers_path = std::source_location::current());
#endif
  //-----------------------sprite sheet animations-----------------------

  std::string read_shader(const std::string& path, const std::source_location& callers_path = std::source_location::current());

  struct light_flags_e {
    enum {
      circle = 0,
      square = 1 << 0,
      lava = 1 << 1, // does this belong here
      additive = 1 << 2,
      multiplicative = 1 << 3,
    };
  };

  template <typename... Ts>
  struct last_sizeof;

  template <typename T>
  struct last_sizeof<T> {
    static constexpr uintptr_t value = sizeof(T);
  };

  template <typename T, typename... Rest>
  struct last_sizeof<T, Rest...> {
    static constexpr uintptr_t value = last_sizeof<Rest...>::value;
  };

  struct shapes {
    struct shape_t;
    
    void shaper_deep_copy(shape_t* dst, const shape_t* const src, shaper_t::ShapeTypeIndex_t sti);

    template<
      typename... Ts,
      uintptr_t s = (sizeof(Ts) + ...)
    >static constexpr shaper_t::ShapeID_t shape_add(
      shaper_t::ShapeTypeIndex_t sti,
      const auto& rd,
      const auto& d,
      Ts... args
    ) {
      struct structarr_t {
        uint8_t p[s];
        uint8_t& operator[](uintptr_t i) {
          return p[i];
        }
      };
      structarr_t a;
      uintptr_t i = 0;
      ([&](auto arg) {
        __builtin_memcpy(&a[i], &arg, sizeof(arg));
        i += sizeof(arg);
        }(args), ...);

      constexpr uintptr_t count = (!!(sizeof(Ts) + 1) + ...);
      static_assert(count % 2 == 0);
      constexpr uintptr_t last_sizeof_v = last_sizeof<Ts...>::value;

      uintptr_t LastKeyOffset = s - last_sizeof_v - 1;
      fan::graphics::g_shapes->shaper.PrepareKeysForAdd(&a, LastKeyOffset);
      return fan::graphics::g_shapes->shaper.add(sti, &a, s, &rd, &d);
    }

    #include "shapes.h"
    using shape_type_t = fan::graphics::shape_type_t;

    // key pack
    struct kp {
      enum {
        light,
        common,
        vfi,
        texture,
      };
    };

    #include <fan/graphics/shape_functions.h>

    shape_functions_t shape_functions;

    struct shape_t : public shaper_t::ShapeID_t {
      using shaper_t::ShapeID_t::ShapeID_t;
      shape_t();
      template <typename T>
        requires requires(T t) { typename T::type_t; }
      shape_t(const T& properties) : shape_t() {
        auto shape_type = T::type_t::shape_type;
        *this = fan::graphics::g_shapes->shape_functions[shape_type].push_back((void*)&properties);
      #if defined(debug_shape_t)
        fan::print("+", NRI);
      #endif
      }
      shape_t(shaper_t::ShapeID_t&& s);
      shape_t(const shaper_t::ShapeID_t& s);
      shape_t(shape_t&& s);
      shape_t(const fan::graphics::shapes::shape_t& s);
      fan::graphics::shapes::shape_t& operator=(const fan::graphics::shapes::shape_t& s);
      fan::graphics::shapes::shape_t& operator=(fan::graphics::shapes::shape_t&& s);
    #if defined(fan_json)
      operator fan::json();
      operator std::string();
      shape_t(const fan::json& json);
      shape_t(const std::string&); // assume json string
      shape_t& operator=(const fan::json& json);
      shape_t& operator=(const std::string&); // assume json string
    #endif
      ~shape_t();
      operator bool() const;
      bool operator==(const shape_t& shape) const;
      void remove();
      void erase();

      shaper_t::ShapeID_t get_id() const;

      bool is_visible() const;
      void set_visible(bool flag);
      void set_static();
      void set_dynamic();

      // many things assume uint16_t so thats why not shaper_t::ShapeTypeIndex_t
      uint16_t get_shape_type() const;
      void set_position(const fan::vec2& position);
      void set_position(const fan::vec3& position);
      void set_x(f32_t x);
      void set_y(f32_t y);
      void set_z(f32_t z);
      fan::vec3 get_position() const;
      f32_t get_x() const;
      f32_t get_y() const;
      f32_t get_z() const;
      void set_size(const fan::vec2& size);
      void set_size3(const fan::vec3& size);
      // returns half extents of draw
      fan::vec2 get_size() const;
      fan::vec3 get_size3();
      void set_rotation_point(const fan::vec2& rotation_point);
      fan::vec2 get_rotation_point() const;
      void set_color(const fan::color& color);
      fan::color get_color() const;
      void set_angle(const fan::vec3& angle);
      fan::vec3 get_angle() const;
      fan::basis get_basis() const;
      fan::vec3 get_forward() const;
      fan::vec3 get_right() const;
      fan::vec3 get_up() const;
      fan::mat3 get_rotation_matrix() const;
      fan::vec3 transform(const fan::vec3& local) const;
      fan::mat4 get_transform() const;
    #if defined(fan_physics)
      fan::physics::aabb_t get_aabb() const;
    #endif
      fan::vec2 get_tc_position() const;
      void set_tc_position(const fan::vec2& tc_position);
      fan::vec2 get_tc_size() const;
      void set_tc_size(const fan::vec2& tc_size);
      bool load_tp(fan::graphics::texture_pack::ti_t* ti);
      fan::graphics::texture_pack::ti_t get_tp() const;
      bool set_tp(fan::graphics::texture_pack::ti_t* ti);
      fan::graphics::texture_pack::unique_t get_tp_unique() const;
      fan::graphics::camera_t get_camera() const;
      void set_camera(fan::graphics::camera_t camera);
      fan::graphics::viewport_t get_viewport() const;
      void set_viewport(fan::graphics::viewport_t viewport);
      render_view_t get_render_view() const;
      void set_render_view(const fan::graphics::render_view_t& render_view);
      fan::vec2 get_grid_size();
      void set_grid_size(const fan::vec2& grid_size);
      fan::graphics::image_t get_image() const;
      void set_image(fan::graphics::image_t image);
      fan::graphics::image_data_t& get_image_data();
      std::array<fan::graphics::image_t, 30> get_images() const;
      void set_images(const std::array<fan::graphics::image_t, 30>& images);
      f32_t get_parallax_factor() const;
      void set_parallax_factor(f32_t parallax_factor);
      uint32_t get_flags() const;
      void set_flags(uint32_t flag);
      f32_t get_radius() const;
      fan::vec3 get_src() const;
      fan::vec3 get_dst() const;
      f32_t get_outline_size() const;
      fan::color get_outline_color() const;
      void set_outline_color(const fan::color& color);
      void reload(uint8_t format, void** image_data, const fan::vec2& image_size);
      void reload(uint8_t format, const fan::vec2& image_size);
      // universal image specific
      void reload(uint8_t format, fan::graphics::image_t images[4]);
      void set_line(const fan::vec2& src, const fan::vec2& dst);
      bool is_mouse_inside();
    #if defined(fan_physics)
      bool intersects(const fan::graphics::shapes::shape_t& shape) const;
      bool collides(const fan::graphics::shapes::shape_t& shape) const;
      bool point_inside(const fan::vec2& point) const;
      bool collides(const fan::vec2& point) const;
    #endif
      // Returns the sign (+1 or −1) of each UV axis, indicating their orientation.
      fan::vec2 get_image_sign() const;
      void add_existing_animation(animation_nr_t nr);
      bool is_animation_finished() const;
      bool is_animation_finished(animation_nr_t nr) const;
      void set_animation_loop(animation_nr_t nr, bool flag);
      void reset_current_sprite_sheet_animation_frame();
      void reset_current_sprite_sheet_animation();
      // sprite sheet - sprite specific
      void set_sprite_sheet_next_frame(int advance = 1);
      animation_shape_nr_t get_shape_animations_id() const;
      std::unordered_map<std::string, fan::graphics::animation_nr_t> get_all_animations() const;
      // Takes in seconds
      void set_sprite_sheet_fps(f32_t fps);
      bool has_animation();
      static void sprite_sheet_frame_update_cb(shaper_t& shaper, shape_t* shape);
      // returns currently active sprite sheet animation
      sprite_sheet_animation_t& get_sprite_sheet_animation();
      void start_sprite_sheet_animation();
      void stop_sprite_sheet_animation();
      // overwrites 'ri.current_animation' animation
      void set_sprite_sheet_animation(const std::string& name);
      void set_sprite_sheet_animation(const sprite_sheet_animation_t& animation);
      void add_sprite_sheet_animation(const sprite_sheet_animation_t& animation);
      void set_sprite_sheet_frames(uint32_t image_index, int horizontal_frames, int vertical_frames);
      animation_nr_t& get_current_animation_id() const;
      bool animation_on(const std::string& name, int frame_index);
      bool animation_on(const std::string& name, const std::initializer_list<int>& arr);
      void set_current_animation_id(animation_nr_t animation_id);
      sprite_sheet_animation_t& get_current_animation();
      int get_current_animation_frame() const;
      void set_current_animation_frame(int frame_id);
      int get_current_animation_frame_count();
      // dont store the pointer
      sprite_sheet_animation_t* get_animation(const std::string& name);
      void set_light_position(const fan::vec3& new_pos);
      void set_light_radius(f32_t radius);
      // for line
      void set_thickness(f32_t new_thickness);
      void apply_floating_motion(f32_t time = 0.f/*start_time.seconds()*/, f32_t amplitude = 5.f, f32_t speed = 2.f, f32_t phase = 0.f);
      void start_particles();
      void stop_particles();

      //vram
      //_d = decltype usage decltype(itself)
      template <typename T>
      T& get_vdata() {
        return *(T*)GetRenderData(g_shapes->shaper);
      }
      template <typename T>
      T& get_data() {
        return *(T*)GetData(g_shapes->shaper);
      }
      // read from gpu itself
      template <typename T>
      T get_gldata() {
        T vi;
        glFinish();
        auto& vbo = get_shape_type_data().renderer.gl.m_vbo;
        get_shape_type_data().renderer.gl.m_vao.bind(*(fan::opengl::context_t*)ctx().render_context);
        vbo.bind(*(fan::opengl::context_t*)ctx().render_context);
        fan::opengl::core::get_glbuffer(*(fan::opengl::context_t*)ctx().render_context, &vi, vbo.m_buffer, sizeof(vi), 0, vbo.m_target);
        return vi;
      }

      

      shaper_t::ShapeTypes_t::nd_t& get_shape_type_data() {
        return g_shapes->shaper.ShapeTypes[get_shape_type()];
      }
      //vram
      template <typename T>
      T::vi_t& get_shape_vdata() {
        return *(typename T::vi_t*)GetRenderData(g_shapes->shaper);
      }
      template <typename T>
      T::ri_t& get_shape_data() {
        return *(typename T::ri_t*)GetData(g_shapes->shaper);
      }
    };

    shaper_t shaper;

    fan::graphics::culling::culling_t visibility;

  #include <fan/graphics/gui/vfi.h>
    vfi_t vfi;

		fan::graphics::texture_pack_t* texture_pack = nullptr;

    struct light_t {

      static inline fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::light;
      static constexpr int kpi = kp::light;

    #pragma pack(push, 1)

      struct vi_t {
        fan::vec3 position;
        f32_t parallax_factor;
        fan::vec2 size;
        fan::vec2 rotation_point;
        fan::color color;
        uint32_t flags = 0;
        fan::vec3 angle;
      };;
      struct ri_t {

      };

    #pragma pack(pop)

      static std::array<shape_gl_init_t, 7>& get_locations() {
        static std::array<shape_gl_init_t, 7> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position)},
            shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, parallax_factor)},
            shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, size)},
            shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, rotation_point)},
            shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color)},
            shape_gl_init_t{{5, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), offsetof(vi_t, flags)},
            shape_gl_init_t{{6, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, angle)}
          }};
        return locs;
      }

      struct properties_t {
        using type_t = light_t;


        fan::vec3 position = 0;
        f32_t parallax_factor = 0;
        fan::vec2 size = 0;
        fan::vec2 rotation_point = 0;
        fan::color color = fan::colors::white;
        uint32_t flags = 0;
        fan::vec3 angle = 0;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }light;

    struct line_t {

      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::line;
      static constexpr int kpi = kp::common;

    #pragma pack(push, 1)

      struct vi_t {
        fan::color color;
        fan::vec3 src;
        fan::vec2 dst;
        f32_t thickness;
        f32_t pad;
      };
      struct ri_t {

      };

    #pragma pack(pop)

      static std::array<shape_gl_init_t, 4>& get_locations() {
        static std::array<shape_gl_init_t, 4> locs{{
            shape_gl_init_t{{0, "in_color"}, decltype(vi_t::color)::size(), GL_FLOAT, sizeof(line_t::vi_t), offsetof(line_t::vi_t, color)},
            shape_gl_init_t{{1, "in_src"}, decltype(vi_t::src)::size(), GL_FLOAT, sizeof(line_t::vi_t), offsetof(line_t::vi_t, src)},
            shape_gl_init_t{{2, "in_dst"}, decltype(vi_t::dst)::size(), GL_FLOAT, sizeof(line_t::vi_t), offsetof(line_t::vi_t, dst)},
            shape_gl_init_t{{3, "line_thickness"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, thickness)}
          }};
        return locs;
      }

      struct properties_t {
        using type_t = line_t;

        fan::vec3 src = 0;
        fan::vec3 dst = 800;
        fan::color color = fan::colors::white;
        f32_t thickness = 4.0f;

        bool blending = true;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }line;

    struct rectangle_t {

      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::rectangle;
      static constexpr int kpi = kp::common;

    #pragma pack(push, 1)

      struct vi_t {
        fan::vec3 position;
        f32_t pad;
        fan::vec2 size;
        fan::vec2 rotation_point;
        fan::color color;
        fan::color outline_color;
        fan::vec3 angle;
        f32_t pad2;
      };
      struct ri_t {

      };

    #pragma pack(pop)

      static std::array<shape_gl_init_t, 6>& get_locations() {
        static std::array<shape_gl_init_t, 6> locs{{
            shape_gl_init_t{{0, "in_position"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position)},
            shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, size)},
            shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, rotation_point)},
            shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color)},
            shape_gl_init_t{{4, "in_outline_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, outline_color)},
            shape_gl_init_t{{5, "in_angle"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, angle)}
          }};
        return locs;
      }

      struct properties_t {
        using type_t = rectangle_t;

        fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::ctx().window->get_size() / 2), 0);
        fan::vec2 size = fan::vec2(32, 32);
        fan::color color = fan::colors::white;
        fan::color outline_color = color;
        bool blending = false;
        fan::vec3 angle = 0;
        fan::vec2 rotation_point = 0;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;
        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }rectangle;

    struct sprite_t {

      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::sprite;
      static constexpr int kpi = kp::texture;

    #pragma pack(push, 1)

      struct vi_t {
        fan::vec3 position;
        f32_t parallax_factor;
        fan::vec2 size;
        fan::vec2 rotation_point;
        fan::color color;
        fan::vec3 angle;
        uint32_t flags;
        fan::vec2 tc_position;
        fan::vec2 tc_size;
        f32_t seed;
        fan::vec3 pad;
      };
      struct ri_t {
        std::array<fan::graphics::image_t, 30> images;
        fan::graphics::texture_pack::unique_t texture_pack_unique_id;

        sprite_sheet_data_t sprite_sheet_data;

        animation_shape_nr_t shape_animations;
        animation_nr_t current_animation;
      };

    #pragma pack(pop)

      static std::array<shape_gl_init_t, 10>& get_locations() {
        static std::array<shape_gl_init_t, 10> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position)},
            shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, parallax_factor)},
            shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, size)},
            shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, rotation_point)},
            shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color)},
            shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, angle)},
            shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), offsetof(vi_t, flags)},
            shape_gl_init_t{{7, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, tc_position)},
            shape_gl_init_t{{8, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, tc_size)},
            shape_gl_init_t{{9, "in_seed"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, seed)},
          }};
        return locs;
      }

      struct properties_t {
        using type_t = sprite_t;

        fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::ctx().window->get_size() / 2), 0);
        f32_t parallax_factor = 0;
        fan::vec2 size = fan::vec2(32, 32);
        fan::vec2 rotation_point = 0;
        fan::color color = fan::colors::white;
        fan::vec3 angle = fan::vec3(0);
        uint32_t flags = light_flags_e::circle | light_flags_e::multiplicative;
        fan::vec2 tc_position = 0;
        fan::vec2 tc_size = 1;
        f32_t seed = 0;
        fan::graphics::texture_pack::unique_t texture_pack_unique_id;
        animation_shape_nr_t shape_animations;
        animation_nr_t current_animation;

        bool load_tp(fan::graphics::texture_pack::ti_t* ti) {
          auto& im = ti->image;
          image = im;
          auto& img = image_get_data(im);
          tc_position = ti->position / img.size;
          tc_size = ti->size / img.size;
          texture_pack_unique_id = ti->unique_id;
          return 0;
        }

        bool blending = false;

        fan::graphics::image_t image = fan::graphics::ctx().default_texture;
        std::array<fan::graphics::image_t, 30> images;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;
        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }sprite;

    struct unlit_sprite_t {

      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::unlit_sprite;
      static constexpr int kpi = kp::texture;

    #pragma pack(push, 1)

      struct vi_t {
        fan::vec3 position;
        f32_t parallax_factor;
        fan::vec2 size;
        fan::vec2 rotation_point;
        fan::color color;
        fan::vec3 angle;
        uint32_t flags;
        fan::vec2 tc_position;
        fan::vec2 tc_size;
        f32_t seed = 0;
      };
      struct ri_t {
        std::array<fan::graphics::image_t, 30> images;
        fan::graphics::texture_pack::unique_t texture_pack_unique_id;
      };

    #pragma pack(pop)

      static std::array<shape_gl_init_t, 10>& get_locations() {
        static std::array<shape_gl_init_t, 10> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position)},
            shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, parallax_factor)},
            shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, size)},
            shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, rotation_point)},
            shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color)},
            shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, angle)},
            shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), offsetof(vi_t, flags)},
            shape_gl_init_t{{7, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, tc_position)},
            shape_gl_init_t{{8, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, tc_size)},
            shape_gl_init_t{{9, "in_seed"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, seed)},
          }};
        return locs;
      }

      struct properties_t {
        using type_t = unlit_sprite_t;

        fan::vec3 position = fan::vec3(fan::vec2(fan::graphics::ctx().window->get_size() / 2), 0);
        f32_t parallax_factor = 0;
        fan::vec2 size = 32;
        fan::vec2 rotation_point = 0;
        fan::color color = fan::colors::white;
        fan::vec3 angle = fan::vec3(0);
        int flags = 0;
        fan::vec2 tc_position = 0;
        fan::vec2 tc_size = 1;
        f32_t seed = 0;

        bool blending = false;

        fan::graphics::image_t image = fan::graphics::ctx().default_texture;
        std::array<fan::graphics::image_t, 30> images;
        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
        fan::graphics::texture_pack::unique_t texture_pack_unique_id;

        bool load_tp(fan::graphics::texture_pack::ti_t* ti) {
          auto& im = ti->image;
          image = im;
          tc_position = ti->position / im.get_size();
          tc_size = ti->size / im.get_size();
          texture_pack_unique_id = ti->unique_id;
          return 0;
        }
      };

      shape_t push_back(const properties_t& properties);

    }unlit_sprite;

    struct text_t {

      struct vi_t {

      };

      struct ri_t {

      };

      struct properties_t {
        using type_t = text_t;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        fan::vec3 position;
        f32_t outline_size = 1;
        fan::vec2 size;
        fan::vec2 tc_position;
        fan::color color = fan::colors::white;
        fan::color outline_color;
        fan::vec2 tc_size;
        fan::vec3 angle = 0;

        std::string text;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }text;

    struct circle_t {

      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::circle;
      static constexpr int kpi = kp::common;

    #pragma pack(push, 1)

      struct vi_t {
        fan::vec3 position;
        f32_t radius;
        fan::vec2 rotation_point;
        fan::color color;
        fan::vec3 angle;
        uint32_t flags;
      };
      struct ri_t {

      };

    #pragma pack(pop)

      static std::array<shape_gl_init_t, 6>& get_locations() {
        static std::array<shape_gl_init_t, 6> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position) },
            shape_gl_init_t{{1, "in_radius"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, radius)},
            shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, rotation_point)},
            shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color)},
            shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, angle)},
            shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), offsetof(vi_t, flags)}
          }};
        return locs;
      }

      struct properties_t {
        using type_t = circle_t;

        fan::vec3 position = 0;
        f32_t radius = 0;
        fan::vec2 rotation_point = 0;
        fan::color color = fan::colors::white;
        fan::vec3 angle = 0;
        uint32_t flags = 0;

        bool blending = false;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };


      fan::graphics::shapes::shape_t push_back(const circle_t::properties_t& properties);

    }circle;

    struct capsule_t {

      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::capsule;
      static constexpr int kpi = kp::common;

    #pragma pack(push, 1)

      struct vi_t {
        fan::vec3 position;
        fan::vec2 center0;
        fan::vec2 center1;
        f32_t radius;
        fan::vec2 rotation_point;
        fan::color color;
        fan::vec3 angle;
        uint32_t flags;
        fan::color outline_color;
      };
      struct ri_t {

      };

    #pragma pack(pop)

      static std::array<shape_gl_init_t, 9>& get_locations() {
        static std::array<shape_gl_init_t, 9> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position) },
            shape_gl_init_t{{1, "in_center0"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, center0)},
            shape_gl_init_t{{2, "in_center1"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, center1)},
            shape_gl_init_t{{3, "in_radius"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, radius)},
            shape_gl_init_t{{4, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, rotation_point)},
            shape_gl_init_t{{5, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color)},
            shape_gl_init_t{{6, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, angle)},
            shape_gl_init_t{{7, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), offsetof(vi_t, flags)},
            shape_gl_init_t{{8, "in_outline_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, outline_color)},
          }};
        return locs;
      }

      struct properties_t {
        using type_t = capsule_t;

        fan::vec3 position = 0;
        fan::vec2 center0 = 0;
        fan::vec2 center1 = { 0, 1.f };
        f32_t radius = 0;
        fan::vec2 rotation_point = 0;
        fan::color color = fan::colors::white;
        fan::color outline_color = color;
        fan::vec3 angle = 0;
        uint32_t flags = 0;

        bool blending = true;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      fan::graphics::shapes::shape_t push_back(const capsule_t::properties_t& properties);
    }capsule;

  #pragma pack(push, 1)

    struct polygon_t {
      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::polygon;
      static constexpr int kpi = kp::common;


      struct vi_t {

      };
      struct ri_t {
        uint32_t buffer_size = 0;
        fan::opengl::core::vao_t vao;
        fan::opengl::core::vbo_t vbo;
      };

    #pragma pack(pop)

      static std::array<shape_gl_init_t, 5>& get_locations() {
        static std::array<shape_gl_init_t, 5> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), offsetof(polygon_vertex_t, position)},
            shape_gl_init_t{{1, "in_color"}, 4, GL_FLOAT, sizeof(polygon_vertex_t), offsetof(polygon_vertex_t, color)},
            shape_gl_init_t{{2, "in_offset"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), offsetof(polygon_vertex_t, offset)},
            shape_gl_init_t{{3, "in_angle"}, 3, GL_FLOAT, sizeof(polygon_vertex_t), offsetof(polygon_vertex_t, angle)},
            shape_gl_init_t{{4, "in_rotation_point"}, 2, GL_FLOAT, sizeof(polygon_vertex_t), offsetof(polygon_vertex_t, rotation_point)},
          }};
        return locs;
      }

      struct properties_t {
        using type_t = polygon_t;
        fan::vec3 position = 0;
        fan::vec3 angle = 0;
        fan::vec2 rotation_point = 0;
        std::vector<vertex_t> vertices;
        bool blending = true;
        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 3;
      };

      fan::graphics::shapes::shape_t push_back(const properties_t& properties);
    }polygon;

    struct grid_t {

      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::grid;
      static constexpr int kpi = kp::common;

    #pragma pack(push, 1)

      struct vi_t {
        fan::vec3 position;
        fan::vec2 size;
        fan::vec2 grid_size;
        fan::vec2 rotation_point;
        fan::color color;
        fan::vec3 angle;
      };
      struct ri_t {

      };

    #pragma pack(pop)

      static std::array<shape_gl_init_t, 6>& get_locations() {
        static std::array<shape_gl_init_t, 6> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position)},
            shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, size)},
            shape_gl_init_t{{2, "in_grid_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, grid_size)},
            shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, rotation_point)},
            shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color)},
            shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, angle)},
          }};
        return locs;
      }

      struct properties_t {
        using type_t = grid_t;

        fan::vec3 position = 0;
        fan::vec2 size = 0;
        fan::vec2 grid_size;
        fan::vec2 rotation_point = 0;
        fan::color color = fan::colors::white;
        fan::vec3 angle = 0;

        bool blending = true;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }grid;


    struct particles_t {

      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::particles;
      static constexpr int kpi = kp::texture;

      static std::array<shape_gl_init_t, 0>& get_locations() {
        static std::array<shape_gl_init_t, 0> locs = {};
        return locs;
      }

    #pragma pack(push, 1)

      struct vi_t {

      };

      struct shapes_e {
        enum {
          circle,
          rectangle
        };
      };

      struct ri_t {

        bool loop = true;
        f32_t loop_enabled_time;
        f32_t loop_disabled_time;

        fan::vec3 position;
        fan::vec2 size;
        fan::color color;

        uint64_t begin_time;
        f32_t alive_time;
        f32_t respawn_time;
        uint32_t count;
        fan::vec2 position_velocity;
        fan::vec3 angle_velocity;
        f32_t begin_angle;
        f32_t end_angle;

        fan::vec3 angle;

        fan::vec2 gap_size;
        f32_t expansion_power;
        fan::vec2 max_spread_size;
        fan::vec2 size_velocity;

        uint32_t shape;

        bool blending;
      };
    #pragma pack(pop)

      struct properties_t {
        using type_t = particles_t;

        bool loop = true;
        f32_t loop_enabled_time = 0.0f;
        f32_t loop_disabled_time = 0.0f;

        fan::vec3 position = 0;
        fan::vec2 size = 100;
        fan::color color = fan::colors::white;

        uint64_t begin_time = 0;
        f32_t alive_time = 1;
        f32_t respawn_time = 0;
        uint32_t count = 10;
        fan::vec2 position_velocity = 130;
        fan::vec3 angle_velocity = fan::vec3(0, 0, 0);
        f32_t begin_angle = 0;
        f32_t end_angle = fan::math::pi * 2;
        fan::vec3 angle = 0;

        fan::vec2 gap_size = 1;
        fan::vec2 max_spread_size = 100;
        f32_t expansion_power = 1.0f;
        fan::vec2 size_velocity = 1;

        uint32_t shape = shapes_e::circle;

        bool blending = true;

        fan::graphics::image_t image = fan::graphics::ctx().default_texture;
        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }particles;
    struct universal_image_renderer_t {
      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::universal_image_renderer;
      static constexpr int kpi = kp::texture;
    #pragma pack(push, 1)
      struct vi_t {
        fan::vec3 position = 0;
        fan::vec2 size = 0;
        fan::vec2 tc_position = 0;
        fan::vec2 tc_size = 1;
      };
      struct ri_t {
        std::array<fan::graphics::image_t, 3> images_rest;
        uint8_t format = fan::graphics::image_format::undefined;
      };
    #pragma pack(pop)
      static std::array<shape_gl_init_t, 4>& get_locations() {
        static std::array<shape_gl_init_t, 4> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position)},
            shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, size)},
            shape_gl_init_t{{2, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, tc_position)},
            shape_gl_init_t{{3, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, tc_size)}
          }};
        return locs;
      }

      struct properties_t {
        using type_t = universal_image_renderer_t;

        fan::vec3 position = 0;
        fan::vec2 size = 0;
        fan::vec2 tc_position = 0;
        fan::vec2 tc_size = 1;

        bool blending = false;

        std::array<fan::graphics::image_t, 4> images = {
          fan::graphics::ctx().default_texture,
          fan::graphics::ctx().default_texture,
          fan::graphics::ctx().default_texture,
          fan::graphics::ctx().default_texture
        };
        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }universal_image_renderer;
    struct gradient_t {
      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::gradient;
      static constexpr int kpi = kp::common;
    #pragma pack(push, 1)
      struct vi_t {
        fan::vec3 position;
        fan::vec2 size;
        fan::vec2 rotation_point;
        std::array<fan::color, 4> color;
        fan::vec3 angle;
      };
      struct ri_t {

      };
    #pragma pack(pop)
      static std::array<shape_gl_init_t, 8>& get_locations() {
        static std::array<shape_gl_init_t, 8> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position)},
            shape_gl_init_t{{1, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, size)},
            shape_gl_init_t{{2, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, rotation_point)},
            shape_gl_init_t{{3, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color) + sizeof(fan::color) * 0},
            shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color) + sizeof(fan::color) * 1},
            shape_gl_init_t{{5, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color) + sizeof(fan::color) * 2},
            shape_gl_init_t{{6, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color) + sizeof(fan::color) * 3},
            shape_gl_init_t{{7, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, angle)}
          }};
        return locs;
      }

      struct properties_t {
        using type_t = gradient_t;

        fan::vec3 position = 0;
        fan::vec2 size = 0;
        std::array<fan::color, 4> color = {
          fan::random::color(),
          fan::random::color(),
          fan::random::color(),
          fan::random::color()
        };
        bool blending = false;
        fan::vec3 angle = 0;
        fan::vec2 rotation_point = 0;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }gradient;
    struct shadow_t {
      static inline fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::shadow;
      static constexpr int kpi = kp::light;
    #pragma pack(push, 1)
      enum shape_e {
        rectangle,
        circle
      };

      struct vi_t {
        fan::vec3 position;
        int shape;
        fan::vec2 size;
        fan::vec2 rotation_point;
        fan::color color;
        uint32_t flags = 0;
        fan::vec3 angle;
        fan::vec2 light_position;
        f32_t light_radius;
        f32_t pad;
      };
      struct ri_t {

      };
    #pragma pack(pop)
      static std::array<shape_gl_init_t, 10>& get_locations() {
        static std::array<shape_gl_init_t, 10> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position)},
            shape_gl_init_t{{1, "in_shape"}, 1, GL_INT, sizeof(vi_t), offsetof(vi_t, shape)},
            shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, size)},
            shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, rotation_point)},
            shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color)},
            shape_gl_init_t{{5, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), offsetof(vi_t, flags)},
            shape_gl_init_t{{6, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, angle)},
            shape_gl_init_t{{7, "in_light_position"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, light_position)},
            shape_gl_init_t{{8, "in_light_radius"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, light_radius)},
            shape_gl_init_t{{9, "in_pad"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, pad)},
          }};
        return locs;
      }

      struct properties_t {
        using type_t = shadow_t;


        fan::vec3 position = 0;
        int shape = shadow_t::rectangle;
        fan::vec2 size = 0;
        fan::vec2 rotation_point = 0;
        fan::color color = fan::colors::white;
        uint32_t flags = 0;
        fan::vec3 angle = 0;
        fan::vec2 light_position = 0;
        f32_t light_radius = 100.f;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }shadow;
    struct shader_shape_t {
      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::shader_shape;
      static constexpr int kpi = kp::texture;
    #pragma pack(push, 1)
      struct vi_t {
        fan::vec3 position;
        f32_t parallax_factor;
        fan::vec2 size;
        fan::vec2 rotation_point;
        fan::color color;
        fan::vec3 angle;
        uint32_t flags;
        fan::vec2 tc_position;
        fan::vec2 tc_size;
        f32_t seed;
      };
      struct ri_t {
        std::array<fan::graphics::image_t, 30> images;
      };
    #pragma pack(pop)
      static std::array<shape_gl_init_t, 10>& get_locations() {
        static std::array<shape_gl_init_t, 10> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, position)},
            shape_gl_init_t{{1, "in_parallax_factor"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, parallax_factor)},
            shape_gl_init_t{{2, "in_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, size)},
            shape_gl_init_t{{3, "in_rotation_point"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, rotation_point)},
            shape_gl_init_t{{4, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color)},
            shape_gl_init_t{{5, "in_angle"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, angle)},
            shape_gl_init_t{{6, "in_flags"}, 1, GL_UNSIGNED_INT , sizeof(vi_t), offsetof(vi_t, flags)},
            shape_gl_init_t{{7, "in_tc_position"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, tc_position)},
            shape_gl_init_t{{8, "in_tc_size"}, 2, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, tc_size)},
            shape_gl_init_t{{9, "in_seed"}, 1, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, seed)},
          }};
        return locs;
      }

      struct properties_t {
        using type_t = shader_shape_t;

        fan::vec3 position = 0;
        f32_t parallax_factor = 0;
        fan::vec2 size = 0;
        fan::vec2 rotation_point = 0;
        fan::color color = fan::colors::white;
        fan::vec3 angle = fan::vec3(0);
        uint32_t flags = 0;
        fan::vec2 tc_position = 0;
        fan::vec2 tc_size = 1;
        f32_t seed = 0;
        fan::graphics::shader_t shader;
        bool blending = true;

        fan::graphics::image_t image = fan::graphics::ctx().default_texture;
        std::array<fan::graphics::image_t, 30> images;

        fan::graphics::camera_t camera = fan::graphics::get_orthographic_render_view().camera;
        fan::graphics::viewport_t viewport = fan::graphics::get_orthographic_render_view().viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 6;
      };

      shape_t push_back(const properties_t& properties);
    }shader_shape;
  #if defined(fan_3D)
    struct rectangle3d_t {
      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::rectangle3d;
      static constexpr int kpi = kp::common;
    #pragma pack(push, 1)
      struct vi_t {
        fan::vec3 position;
        fan::vec3 size;
        fan::color color;
        fan::vec3 angle;
      };
      struct ri_t {

      };
    #pragma pack(pop)
      static std::array<shape_gl_init_t, 3>& get_locations() {
        static std::array<shape_gl_init_t, 3> locs{{
            shape_gl_init_t{{0, "in_position"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t,  position)},
            shape_gl_init_t{{1, "in_size"}, 3, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, size)},
            shape_gl_init_t{{2, "in_color"}, 4, GL_FLOAT, sizeof(vi_t), offsetof(vi_t, color)}
          }};
        return locs;
      }

      struct properties_t {
        using type_t = rectangle_t;

        fan::vec3 position = 0;
        fan::vec3 size = 0;
        fan::color color = fan::colors::white;
        bool blending = false;
        fan::vec3 angle = 0;

        fan::graphics::camera_t camera = fan::graphics::ctx().perspective_render_view->camera;
        fan::graphics::viewport_t viewport = fan::graphics::ctx().perspective_render_view->viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::triangles;
        uint32_t vertex_count = 36;
      };


      shape_t push_back(const properties_t& properties);
    }rectangle3d;
    struct line3d_t {
      static constexpr fan::graphics::shaper_t::KeyTypeIndex_t shape_type = shape_type_t::line3d;
      static constexpr int kpi = kp::common;
    #pragma pack(push, 1)
      struct vi_t {
        fan::color color;
        fan::vec3 src;
        fan::vec3 dst;
      };
      struct ri_t {

      };
    #pragma pack(pop)
      static std::array<shape_gl_init_t, 3>& get_locations() {
        static std::array<shape_gl_init_t, 3> locs{{
            shape_gl_init_t{{0, "in_color"}, 4, GL_FLOAT, sizeof(line_t::vi_t), offsetof(line_t::vi_t, color)},
            shape_gl_init_t{{1, "in_src"}, 3, GL_FLOAT, sizeof(line_t::vi_t), offsetof(line_t::vi_t, src)},
            shape_gl_init_t{{2, "in_dst"}, 3, GL_FLOAT, sizeof(line_t::vi_t), offsetof(line_t::vi_t, dst)}
          }};
        return locs;
      }

      struct properties_t {
        using type_t = line_t;

        fan::color color = fan::colors::white;
        fan::vec3 src;
        fan::vec3 dst;

        bool blending = false;

        fan::graphics::camera_t camera = fan::graphics::ctx().perspective_render_view->camera;
        fan::graphics::viewport_t viewport = fan::graphics::ctx().perspective_render_view->viewport;

        uint8_t draw_mode = fan::graphics::primitive_topology_t::lines;
        uint32_t vertex_count = 2;
      };

      shape_t push_back(const properties_t& properties);
    }line3d;
  #endif

    std::vector<fan::graphics::shapes::shape_t>* immediate_render_list = nullptr;
    std::unordered_map<uint32_t, fan::graphics::shapes::shape_t>* static_render_list = nullptr;
	};

  fan::graphics::shapes& get_shapes() {
    return *g_shapes;
  }
}

export namespace fan::graphics {
  struct shape_deserialize_t {
    struct {
      // json::iterator doesnt support union
      // i dont want to use variant either so i accept few extra bytes
    #if defined(fan_json)
      json::const_iterator it;
    #endif
      uint64_t offset = 0;
    }data;
    bool init = false;
    bool was_object = false;
  #if defined(fan_json)
    bool iterate(const fan::json& json, fan::graphics::shapes::shape_t* shape, const std::source_location& callers_path = std::source_location::current());
  #endif
    bool iterate(const std::vector<uint8_t>& bin_data, fan::graphics::shapes::shape_t* shape);
  };

  bool shape_to_bin(fan::graphics::shapes::shape_t& shape, std::vector<uint8_t>* data);
  bool bin_to_shape(const std::vector<uint8_t>& in, fan::graphics::shapes::shape_t* shape, uint64_t& offset, const std::source_location& callers_path = std::source_location::current());
  bool shape_serialize(fan::graphics::shapes::shape_t& shape, std::vector<uint8_t>* out);
}

#if defined(fan_json)
export namespace fan::graphics {
  bool shape_to_json(fan::graphics::shapes::shape_t& shape, fan::json* json);
  bool json_to_shape(const fan::json& in, fan::graphics::shapes::shape_t* shape, const std::source_location& callers_path = std::source_location::current());
  bool shape_serialize(fan::graphics::shapes::shape_t& shape, fan::json* out);
}

export namespace fan::graphics {
  fan::graphics::shapes::shape_t extract_single_shape(const fan::json& json_data, const std::source_location& callers_path = std::source_location::current());
  fan::json read_json(const std::string& path, const std::source_location& callers_path = std::source_location::current());
	struct animation_t {
		fan::graphics::animation_nr_t nr;
	};
	// for dme type
	void map_animations(auto& anims) {
		for (auto [i, animation] : fan::enumerate(fan::graphics::all_animations)) {
			for (int j = 0; j < anims.size(); ++j) {
				auto& anim = *anims.NA(j);
				if (animation.second.name == (const char*)anim) {
					anim = animation_t{ .nr = animation.first };
					break;
				}
			}
		}
	}
} // namespace fan::graphics

#endif