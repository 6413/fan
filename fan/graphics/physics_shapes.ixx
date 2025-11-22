module;

// for shapes

#if defined(fan_physics)
  #include <fan/utility.h>
  #include <box2d/box2d.h>
  #include <fan/types/bll_raii.h>
#endif

#include <source_location>

export module fan.graphics.physics_shapes;

#if defined(fan_physics)

import fan.types.vector;
import fan.types.color;
import fan.time;
import fan.utility;
import fan.physics.b2_integration;
import fan.physics.common_context;
import fan.window.input_action;
import fan.types.json;
import fan.math;

#if (fan_gui)
  import fan.graphics;
  import fan.graphics.common_context;
  import fan.graphics.shapes;
#else
  import fan.graphics;
#endif

// could add debug_draw_render_view for custom view

/// Draw a closed polygon provided in CCW order.
void DrawPolygon(const fan::vec2* vertices, int vertexCount, b2HexColor color, void* context);

/// Draw a solid closed polygon provided in CCW order.
void DrawSolidPolygon(b2Transform transform, const b2Vec2* vertices, int vertexCount, f32_t radius, b2HexColor color,
  void* context);

/// Draw a circle.
void DrawCircle(b2Vec2 center, f32_t radius, b2HexColor color, void* context);

/// Draw a solid circle.
void DrawSolidCircle(b2Transform transform, f32_t radius, b2HexColor color, void* context);

/// Draw a capsule.
void DrawCapsule(b2Vec2 p1, b2Vec2 p2, f32_t radius, b2HexColor color, void* context);

/// Draw a solid capsule.
void DrawSolidCapsule(b2Vec2 p1, b2Vec2 p2, f32_t radius, b2HexColor color, void* context);

/// Draw a line segment.
void DrawSegment(b2Vec2 p1, b2Vec2 p2, b2HexColor color, void* context);

/// Draw a transform. Choose your own length scale.
void DrawTransform(b2Transform transform, void* context);

/// Draw a point.
void DrawPoint(b2Vec2 p, f32_t size, b2HexColor color, void* context);

/// Draw a string.
void DrawString(b2Vec2 p, const char* s, b2HexColor color, void* context);

b2DebugDraw initialize_debug(bool enabled);


export namespace fan {
  namespace graphics {
    namespace physics {

      b2DebugDraw box2d_debug_draw{};

      void init();

      void step(f32_t dt);

      void debug_draw(bool enabled);
      // position & aabb & angle
      std::function<void(fan::graphics::shape_t&, const fan::vec3&, const fan::vec2&, f32_t)> physics_update_cb =
        [](fan::graphics::shape_t&, const fan::vec3&, const fan::vec2&, f32_t) {};

      void shape_physics_update(const fan::physics::physics_update_data_t& data);

      struct mass_data_t {
        // kgs
        f32_t mass = -1.f;
        fan::vec2 center_of_mass = 0.f;
        f32_t rotational_inertia = -1.f;
        operator b2MassData() const;
      };

      struct base_shape_t : fan::graphics::shape_t, fan::physics::entity_t {
        base_shape_t() = default;

        void set_shape(fan::graphics::shape_t&& shape);
        base_shape_t(fan::graphics::shape_t&& shape, fan::physics::entity_t&& entity, const mass_data_t& mass_data);
        base_shape_t(const base_shape_t& r);
        base_shape_t(base_shape_t&& r);
        ~base_shape_t();
        base_shape_t& operator=(const base_shape_t& r);
        base_shape_t& operator=(base_shape_t&& r);

        void erase();

        mass_data_t get_mass_data() const;

        f32_t get_mass() const;

        void set_draw_offset(fan::vec2 new_draw_offset);

        fan::vec3 get_position() const;

        fan::vec2 draw_offset = 0;
        fan::physics::physics_update_cbs_t::nr_t physics_update_nr;
      };

      struct rectangle_t : base_shape_t {
        struct properties_t {
          render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
          fan::vec3 position = fan::vec3(fan::vec2(fan::window::get_size() / 2), 0);
          fan::vec2 size = fan::vec2(32, 32);
          fan::color color = fan::color(1, 1, 1, 1);
          fan::color outline_color = color;
          fan::vec3 angle = 0;
          fan::vec2 rotation_point = 0;
          bool blending = false;
          operator fan::graphics::rectangle_properties_t() const;
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        rectangle_t() = default;
        rectangle_t(const properties_t& p);
        rectangle_t(const rectangle_t& r);
        rectangle_t(rectangle_t&& r) : base_shape_t(std::move(r)) {}
        rectangle_t& operator=(const rectangle_t& r);
        rectangle_t& operator=(rectangle_t&& r);
      };

      struct sprite_t : base_shape_t {
        struct properties_t {
          render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
          fan::vec3 position = fan::vec3(0, 0, 0);
          fan::vec2 size = fan::vec2(0.1, 0.1);
          fan::vec3 angle = 0;
          fan::color color = fan::color(1, 1, 1, 1);
          fan::vec2 rotation_point = 0;
          fan::graphics::image_t image = fan::graphics::get_default_texture();
          std::array<fan::graphics::image_t, 30> images;
          f32_t parallax_factor = 0;
          bool blending = true;
          uint32_t flags = light_flags_e::circle | light_flags_e::multiplicative;
          operator fan::graphics::sprite_properties_t() const {
            return fan::graphics::sprite_properties_t{
              .render_view = render_view,
              .position = position,
              .size = size,
              .angle = angle,
              .color = color,
              .rotation_point = rotation_point,
              .image = image,
              .images = images,
              .parallax_factor = parallax_factor,
              .blending = blending,
              .flags = flags
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        sprite_t() = default;
        sprite_t(const properties_t& p);
        sprite_t(const sprite_t& r);
        sprite_t(sprite_t&& r);
        sprite_t& operator=(const sprite_t& r);
        sprite_t& operator=(sprite_t&& r);
      };

      struct circle_t : base_shape_t {
        struct properties_t {
          render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
          fan::vec3 position = fan::vec3(0, 0, 0);
          f32_t radius = 0.1f;
          fan::vec3 angle = 0;
          fan::color color = fan::color(1, 1, 1, 1);
          bool blending = true;
          uint32_t flags = 0;
          operator fan::graphics::circle_properties_t() const {
            return fan::graphics::circle_properties_t{
              .render_view = render_view,
              .position = position,
              .radius = radius,
              .angle = angle,
              .color = color,
              .blending = blending,
              .flags = flags
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        circle_t() = default;
        circle_t(const properties_t& p);
        circle_t(const circle_t& r);
        circle_t(circle_t&& r);
        circle_t& operator=(const circle_t& r);
        circle_t& operator=(circle_t&& r);
      };

      struct circle_sprite_t : base_shape_t {
        struct properties_t {
          render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
          fan::vec3 position = fan::vec3(0, 0, 0);
          f32_t radius = 0.1f;
          fan::vec2 size = radius;
          fan::vec3 angle = 0;
          fan::graphics::image_t image = fan::graphics::get_default_texture();
          fan::color color = fan::color(1, 1, 1, 1);
          bool blending = true;
          uint32_t flags = light_flags_e::circle | light_flags_e::multiplicative;
          operator fan::graphics::sprite_properties_t() const {
            return fan::graphics::sprite_properties_t{
              .render_view = render_view,
              .position = position,
              .size = size,
              .angle = angle,
              .color = color,
              .image = image,
              .blending = blending,
              .flags = flags
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        circle_sprite_t() = default;
        circle_sprite_t(const properties_t& p);
        circle_sprite_t(const circle_sprite_t& r);
        circle_sprite_t(circle_sprite_t&& r);
        circle_sprite_t& operator=(const circle_sprite_t& r);
        circle_sprite_t& operator=(circle_sprite_t&& r);
      };

      struct capsule_t : base_shape_t {
        struct properties_t {
          render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
          fan::vec3 position = fan::vec3(fan::vec2(fan::window::get_size() / 2), 0);
          fan::vec2 center0{ 0, -32.f };
          fan::vec2 center1{ 0, 32.f };
          f32_t radius = 16.f;
          fan::vec3 angle = 0.f;
          fan::color color = fan::color(1, 1, 1, 1);
          fan::color outline_color = color;
          bool blending = true;
          uint32_t flags = 0;
          operator fan::graphics::capsule_properties_t() const {
            return fan::graphics::capsule_properties_t{
              .render_view = render_view,
              .position = position,
              .center0 = center0,
              .center1 = center1,
              .radius = radius,
              .angle = angle,
              .color = color,
              .outline_color = outline_color,
              .blending = blending,
              .flags = flags
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        capsule_t() = default;
        capsule_t(const properties_t& p);
        capsule_t(const capsule_t& r);
        capsule_t(capsule_t&& r);
        capsule_t& operator=(const capsule_t& r);
        capsule_t& operator=(capsule_t&& r);
      };

      struct capsule_sprite_t : base_shape_t {
        struct properties_t {
          render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
          fan::vec3 position = fan::vec3(0, 0, 0);
          fan::vec2 center0{ 0, -32.f };
          fan::vec2 center1{ 0, 32.f };
          fan::vec2 size = 64.0f;
          fan::vec3 angle = 0;
          fan::color color = fan::color(1, 1, 1, 1);
          fan::vec2 aabb_scale = 1.0;
          fan::vec2 rotation_point = 0;
          fan::graphics::image_t image = fan::graphics::get_default_texture();
          std::array<fan::graphics::image_t, 30> images;
          f32_t parallax_factor = 0;
          bool blending = true;
          uint32_t flags = light_flags_e::circle | light_flags_e::multiplicative;

          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;

          operator fan::graphics::sprite_properties_t() const {
            return fan::graphics::sprite_properties_t{
              .render_view = render_view,
              .position = position,
              .size = size,
              .angle = angle,
              .color = color,
              .rotation_point = rotation_point,
              .image = image,
              .images = images,
              .parallax_factor = parallax_factor,
              .blending = blending,
              .flags = flags
            };
          }
        };
        capsule_sprite_t() = default;
        capsule_sprite_t(const properties_t& p);
        capsule_sprite_t(const capsule_sprite_t& r);
        capsule_sprite_t(capsule_sprite_t&& r);
        capsule_sprite_t& operator=(const capsule_sprite_t& r);
        capsule_sprite_t& operator=(capsule_sprite_t&& r);
      };

      struct polygon_t : base_shape_t {
        struct properties_t {
          render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
          fan::vec3 position = 0;
          f32_t radius = 0.005;
          fan::vec3 angle = 0;
          fan::vec2 rotation_point = 0;
          std::vector<vertex_t> vertices;
          bool blending = true;
          uint8_t draw_mode = fan::graphics::primitive_topology_t::triangle_fan;
          operator fan::graphics::polygon_properties_t() const {
            return fan::graphics::polygon_properties_t{
              .render_view = render_view,
              .position = position,
              .vertices = vertices,
              .angle = angle,
              .rotation_point = rotation_point,
              .blending = blending,
              .draw_mode = draw_mode
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        polygon_t() = default;
        polygon_t(const properties_t& p);
        polygon_t(const polygon_t& r);
        polygon_t(polygon_t&& r);
        polygon_t& operator=(const polygon_t& r);
        polygon_t& operator=(polygon_t&& r);
      };

      struct polygon_strip_t : base_shape_t {
        struct properties_t {
          render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
          fan::vec3 position = 0;
          fan::vec3 angle = 0;
          fan::vec2 rotation_point = 0;
          std::vector<vertex_t> vertices;
          bool blending = true;
          uint8_t draw_mode = fan::graphics::primitive_topology_t::triangle_strip;
          operator fan::graphics::polygon_properties_t() const {
            return fan::graphics::polygon_properties_t{
              .render_view = render_view,
              .position = position,
              .vertices = vertices,
              .angle = angle,
              .rotation_point = rotation_point,
              .blending = blending,
              .draw_mode = draw_mode
            };
          }
          uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        polygon_strip_t() = default;
        polygon_strip_t(const properties_t& p);
        polygon_strip_t(const polygon_strip_t& r);
        polygon_strip_t(polygon_strip_t&& r);
        polygon_strip_t& operator=(const polygon_strip_t& r);
        polygon_strip_t& operator=(polygon_strip_t&& r);
      };

      std::array<fan::graphics::physics::rectangle_t, 4> create_stroked_rectangle(
        const fan::vec2& center_position,
        const fan::vec2& half_size,
        f32_t thickness,
        const fan::color& wall_color = fan::color::from_rgba(0x6e8d6eff),
        std::array<fan::physics::shape_properties_t, 4> shape_properties = { {
          {.friction = 0  },
          {.friction = 0.6},
          {.friction = 0  },
          {.friction = 0  }
        } }
      );
      std::array<rectangle_t, 4> create_walls(
        const fan::vec2& bounds,
        f32_t thickness,
        const fan::color& wall_color = fan::color::from_rgba(0x6e8d6eff),
        std::array<fan::physics::shape_properties_t, 4> shape_properties = { {
          {.friction = 0  },  // top
          {.friction = 0.6},  // bottom
          {.friction = 0  },  // left
          {.friction = 0  }   // right
        } }
      );
      struct character2d_t : physics::base_shape_t {
        using physics::base_shape_t::base_shape_t;

        using movement_callback_handle_t = bll_nr_t<fan::physics::physics_step_callback_nr_t, character2d_t>;

        struct movement_e {
          enum {
            side_view, // left, right, space to jump
            top_view   // left, right, down, up wasd
          };
        };

        struct wall_jump_t {
          fan::vec2 normal;
          f32_t slide_speed = 200.f;
          f32_t push_away_force = 1.f;
        } wall_jump;

        character2d_t() = default;
        character2d_t(auto&& shape) : base_shape_t(std::move(shape)) {}

        character2d_t(const character2d_t& o);
        character2d_t(character2d_t&& o) noexcept;
        character2d_t& operator=(const character2d_t& o);
        character2d_t& operator=(character2d_t&& o) noexcept;

        void set_shape(fan::graphics::shape_t&& shape);
        void set_physics_body(fan::physics::entity_t&& entity);
        void update_animation();
        static bool is_on_ground(fan::physics::body_id_t main, std::array<fan::physics::body_id_t, 2> feet, bool jumping);
        void process_movement(uint8_t movement = movement_e::side_view, f32_t friction = 12);
        void move_to_direction(const fan::vec2& direction);
        void set_physics_position(const fan::vec2& p);
        void enable_default_movement(uint8_t movement = movement_e::side_view);
        void update_animations();
        void setup_default_animations();
        struct animation_controller_t {
          struct animation_state_t {
            fan::graphics::animation_nr_t animation_id;
            int fps = 15;
            bool velocity_based_fps = false;
            std::function<bool(character2d_t&)> condition;
          };

          void add_state(const std::string& name, const animation_state_t& state);
          void update(character2d_t& character);
          std::unordered_map<std::string, animation_state_t> states;
        } anim_controller;

        struct character_config_t {
          std::string json_path;
          f32_t aabb_scale = 1.0f;
          uint8_t movement_type = character2d_t::movement_e::side_view;
          fan::vec2 draw_offset_override = { 0,0 };
          bool auto_draw_offset = true;
          bool auto_animations = true;
          fan::physics::shape_properties_t physics_properties = { .fixed_rotation = true };
        };

        static character2d_t from_json(const character_config_t& config, const std::source_location& callers_path = std::source_location::current());

        movement_callback_handle_t add_movement_callback(std::function<void(character2d_t*)> fn);

        fan::vec2 previous_movement_sign = 0;
        f32_t force = 120.f;
        f32_t jump_impulse = 75.f;
        f32_t max_speed = 600.f;
        f32_t jump_delay = 0.25f;
        bool jumping = false;
        fan::physics::body_id_t feet[2];
        f32_t coyote_time = 0.1f;
        f32_t last_ground_time = 0.f;
        bool handle_jump = true;
        bool on_air_after_jump = false;
        bool jump_consumed = false;
        bool movement_enabled = false;
        bool current_animation_requires_velocity_fps = false;
        bool auto_update_animations = false;
        uint8_t movement_type = movement_e::side_view;
        movement_callback_handle_t movement_cb;
      };

      struct bone_e {
        enum {
          hip = 0,
          torso = 1,
          head = 2,
          upper_left_leg = 3,
          lower_left_leg = 4,
          upper_right_leg = 5,
          lower_right_leg = 6,
          upper_left_arm = 7,
          lower_left_arm = 8,
          upper_right_arm = 9,
          lower_right_arm = 10,
          bone_count = 11,
        };
      };
      constexpr const char* bone_names[] = {
        "Hip", "Torso", "Head",
        "Upper Left Leg", "Lower Left Leg",
        "Upper Right Leg", "Lower Right Leg",
        "Upper Left Arm", "Lower Left Arm",
        "Upper Right Arm", "Lower Right Arm"
      };
      inline std::string bone_to_string(int bone) {
        if (bone >= std::size(bone_names)) {
          return "N/A";
        }
        return bone_names[bone];
      }

      struct bone_t {
        fan::graphics::physics::base_shape_t visual;
        fan::physics::joint_id_t joint_id = b2_nullJointId;
        f32_t friction_scale;
        int parent_index;
        // local
        fan::vec3 position = 0;
        fan::vec2 size = 1;
        fan::vec2 pivot = 0;
        fan::vec2 offset = 0;
        f32_t scale = 1;
        f32_t lower_angle = 0;
        f32_t upper_angle = 0;
        f32_t reference_angle = 0;
        fan::vec2 center0 = 0;
        fan::vec2 center1 = 0;
      };

      void update_reference_angle(b2WorldId world, fan::physics::joint_id_t& joint_id, f32_t new_reference_angle);

      struct human_t {
        using bone_images_t = std::array<fan::graphics::image_t, bone_e::bone_count>;
        using bones_t = std::array<bone_t, bone_e::bone_count>;

        human_t() = default;
        human_t(const fan::vec2& position, const f32_t scale = 1.f, const bone_images_t& images = {}, const fan::color& color = fan::colors::white);

        static void load_bones(const fan::vec2& position, f32_t scale, std::array<fan::graphics::physics::bone_t, fan::graphics::physics::bone_e::bone_count>& bones);

        static bone_images_t load_character_images(const std::string& character_folder_path, const fan::graphics::image_load_properties_t& lp);
        void animate_walk(f32_t force, f32_t dt);

        void load_preset(const fan::vec2& position, const f32_t scale, const bone_images_t& images, std::array<bone_t, bone_e::bone_count>& bones, const fan::color& color = fan::colors::white);
        void load(const fan::vec2& position, const f32_t scale = 1.f, const bone_images_t& images = {}, const fan::color& color = fan::colors::white);

        void animate_jump(f32_t jump_impulse, f32_t dt, bool is_jumping);

        void erase();

        bones_t bones;
        f32_t scale = 1.f;
        bool is_spawned = false;
        int direction = 1;
        int look_direction = direction;
        int go_up = 0;
        fan::time::timer jump_animation_timer;
      };

      struct mouse_joint_t {
        fan::physics::body_id_t dummy_body;
        fan::graphics::update_callback_nr_t nr;

        operator fan::physics::body_id_t& ();
        operator const fan::physics::body_id_t& () const;

        struct QueryContext {
          b2Vec2 point;
          b2BodyId bodyId = b2_nullBodyId;
        };

        static bool QueryCallback(b2ShapeId shapeId, void* context);

        mouse_joint_t();
        ~mouse_joint_t();

        fan::physics::body_id_t target_body;
        fan::physics::joint_id_t mouse_joint = b2_nullJointId;
      };
    }
  }
}

export namespace fan::physics {
  using mouse_joint_t = fan::graphics::physics::mouse_joint_t;
  bool is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id);
  fan::physics::ray_result_t raycast(const fan::vec2& src, const fan::vec2& dst);
}

export namespace fan::graphics::physics {
  using shape_t = fan::graphics::physics::base_shape_t;

  fan::graphics::physics::character2d_t character_circle(
    const fan::vec3& position,
    f32_t radius = 16.f,
    const fan::physics::shape_properties_t& shape_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_circle(
    const fan::graphics::physics::circle_t::properties_t& visual_properties,
    const fan::physics::shape_properties_t& physics_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_circle_sprite(
    const fan::vec3& position,
    f32_t radius = 16.f,
    const fan::graphics::image_t& image = fan::graphics::get_default_texture(),
    const fan::physics::shape_properties_t& shape_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_circle_sprite(
    const fan::graphics::physics::circle_sprite_t::properties_t& visual_properties,
    const fan::physics::shape_properties_t& physics_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_capsule(
    const fan::vec3& position,
    const fan::vec2& center0 = { 0, -32.f },
    const fan::vec2& center1 = { 0, 32.f },
    f32_t radius = 16.f,
    const fan::physics::shape_properties_t& shape_properties = {}
  );
  // creates physics body for visual shape
  fan::physics::entity_t character_capsule(
    const fan::graphics::shape_t& shape,
    f32_t shape_size_multiplier = 1.0,
    const fan::physics::shape_properties_t& physics_properties = { 
      .fixed_rotation = true
    }, uint8_t body_type = fan::physics::body_type_e::dynamic_body);
  fan::graphics::physics::character2d_t character_capsule(
    const fan::graphics::physics::capsule_t::properties_t& visual_properties,
    const fan::physics::shape_properties_t& physics_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_capsule_sprite(
    const fan::vec3& position,
    const fan::vec2& center0 = { 0, -32.f },
    const fan::vec2& center1 = { 0, 32.f },
    const fan::vec2& size = { 64.f, 64.f },
    const fan::graphics::image_t& image = fan::graphics::get_default_texture(),
    const fan::physics::shape_properties_t& shape_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_capsule_sprite(
    const fan::graphics::physics::capsule_sprite_t::properties_t& visual_properties,
    const fan::physics::shape_properties_t& physics_properties = {.fixed_rotation = true}
  );

  fan::graphics::physics::character2d_t character_rectangle(
    const fan::vec3& position,
    const fan::vec2& size = { 32.f, 32.f },
    const fan::physics::shape_properties_t& shape_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_rectangle(
    const fan::graphics::physics::rectangle_t::properties_t& visual_properties,
    const fan::physics::shape_properties_t& physics_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_sprite(
    const fan::vec3& position,
    const fan::vec2& size = { 32.f, 32.f },
    const fan::graphics::image_t& image = fan::graphics::get_default_texture(),
    const fan::physics::shape_properties_t& shape_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_sprite(
    const fan::graphics::physics::sprite_t::properties_t& visual_properties,
    const fan::physics::shape_properties_t& physics_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_polygon(
    const fan::vec3& position,
    const std::vector<fan::graphics::vertex_t>& vertices,
    f32_t radius = 0.005f,
    const fan::physics::shape_properties_t& shape_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_polygon(
    const fan::graphics::physics::polygon_t::properties_t& visual_properties,
    const fan::physics::shape_properties_t& physics_properties = {.fixed_rotation = true}
  );
}

#endif