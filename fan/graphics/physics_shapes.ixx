module;

#if defined (FAN_WINDOW)

// for shapes
#if defined(FAN_2D)
#if defined(FAN_PHYSICS_2D)
  #include <box2d/box2d.h>
#endif
#endif

#endif

export module fan.graphics.physics_shapes;

#if defined (FAN_WINDOW)

import std;

#if defined(FAN_2D)

#if defined(FAN_PHYSICS_2D)

import fan.types;
import fan.types.vector;
import fan.types.color;
import fan.time;
import fan.utility;
import fan.physics.types;
import fan.physics.b2_integration;
import fan.physics.common_context;
import fan.graphics.shapes.types;
import fan.window.input_action;
#if defined(FAN_JSON)
import fan.types.json;
#endif
import fan.math;
import fan.random;
import fan.ecs;

import fan.graphics;
import fan.graphics.common_context;
import fan.graphics.shapes;

import fan.tween;

export namespace fan {
  namespace graphics {
    namespace physics {
      struct character2d_t;

      void init();

      void step(f32_t dt);

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
        base_shape_t(fan::graphics::shape_t&& shape, fan::physics::entity_t&& entity);
        base_shape_t(fan::graphics::shape_t&& shape, fan::physics::entity_t&& entity, const mass_data_t& mass_data);
        base_shape_t(const base_shape_t& r);
        base_shape_t(base_shape_t&& r);
        ~base_shape_t();
        base_shape_t& operator=(const base_shape_t& r);
        base_shape_t& operator=(base_shape_t&& r);

        void erase();

        fan::vec2 get_draw_offset() const;
        void set_draw_offset(fan::vec2 new_draw_offset);

        // set whether to sync angle given by collisions or manually adjust it visually 
        void sync_visual_angle(bool flag);

        fan::vec3 get_position() const;
        fan::vec3 get_physics_position() const;
        fan::vec2 get_size() const;
        fan::physics::aabb_t get_aabb() const;

        fan::physics::collision_listener_handle_t on_collision_enter(std::function<void(fan::physics::entity_t other)> cb);
        fan::physics::collision_listener_handle_t on_collision_exit(std::function<void(fan::physics::entity_t other)> cb);
        void on_sensor_enter(fan::physics::entity_t& target, std::function<void()> callback); // backward compat

        fan::vec2 draw_offset = 0;
        fan::physics::physics_update_cbs_t::nr_t physics_update_nr;
      };

      struct collision_scope_t {
        collision_scope_t() = default;
        ~collision_scope_t();
        collision_scope_t(const collision_scope_t&) = delete;
        collision_scope_t& operator=(const collision_scope_t&) = delete;
        fan::physics::collision_listener_handle_t on_enter(fan::physics::body_id_t body, auto cb) {
          auto h = fan::physics::add_collision_listeners(body, {.on_enter = std::move(cb)});
          handles.push_back(h);
          return h;
        }
        fan::physics::collision_listener_handle_t on_exit(fan::physics::body_id_t body, auto cb) {
          auto h = fan::physics::add_collision_listeners(body, {.on_exit = std::move(cb)});
          handles.push_back(h);
          return h;
        }
        fan::physics::collision_listener_handle_t on_hit(fan::physics::body_id_t body, auto cb) {
          auto h = fan::physics::add_collision_listeners(body, {.on_hit = std::move(cb)});
          handles.push_back(h);
          return h;
        }
        std::vector<fan::physics::collision_listener_handle_t> handles;
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
          std::uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        rectangle_t() = default;
        rectangle_t(const properties_t& p);
        rectangle_t(fan::vec3 position, fan::vec2 size, fan::color color, std::uint8_t body_type);
        rectangle_t(const rectangle_t& r);
        rectangle_t(rectangle_t&& r);
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
          fan::vec2 tc_position = 0;
          fan::vec2 tc_size = 1;
          std::array<fan::graphics::image_t, 30> images;
          fan::vec2 parallax_factor = 0;
          bool blending = true;
          std::uint32_t flags = sprite_flags_e::circle | sprite_flags_e::multiplicative;
          operator fan::graphics::sprite_properties_t() const;
          std::uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        using base_shape_t::base_shape_t;
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
          fan::color color = fan::color(1, 1, 1, 1);
          fan::color outline_color = color;
          f32_t outline_width = shapes::circle_t::properties_t().outline_width;
          fan::vec3 angle = 0;
          bool blending = true;
          std::uint32_t flags = 0;
          operator fan::graphics::circle_properties_t() const;
          std::uint8_t body_type = fan::physics::body_type_e::static_body;
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
          std::uint32_t flags = sprite_flags_e::circle | sprite_flags_e::multiplicative;
          operator fan::graphics::sprite_properties_t() const;
          std::uint8_t body_type = fan::physics::body_type_e::static_body;
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
          fan::vec2 center0{ 0.f, -32.f };
          fan::vec2 center1{ 0.f, 32.f };
          f32_t radius = 16.f;
          fan::vec3 angle = 0.f;
          fan::color color = fan::color(1, 1, 1, 1);
          fan::color outline_color = color;
          bool blending = true;
          std::uint32_t flags = 0;
          operator fan::graphics::capsule_properties_t() const;
          std::uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;
        };
        capsule_t() = default;
        capsule_t(const properties_t& p);
        capsule_t(fan::vec3 position, fan::vec2 center0, fan::vec2 center1, f32_t radius, fan::color color, std::uint8_t body_type);
        capsule_t(const capsule_t& r);
        capsule_t(capsule_t&& r);
        capsule_t& operator=(const capsule_t& r);
        capsule_t& operator=(capsule_t&& r);
      };

      struct capsule_sprite_t : base_shape_t {
        struct properties_t {
          render_view_t* render_view = &fan::graphics::get_orthographic_render_view();
          fan::vec3 position = fan::vec3(0, 0, 0);
          fan::vec2 center0{ 0.f, -32.f };
          fan::vec2 center1{ 0.f, 32.f };
          fan::vec2 size = 64.0f;
          fan::vec3 angle = 0;
          fan::color color = fan::color(1, 1, 1, 1);
          fan::vec2 aabb_scale = 1.0;
          fan::vec2 rotation_point = 0;
          fan::graphics::image_t image = fan::graphics::get_default_texture();
          std::array<fan::graphics::image_t, 30> images;
          fan::vec2 parallax_factor = 0;
          bool blending = true;
          std::uint32_t flags = sprite_flags_e::circle | sprite_flags_e::multiplicative;

          std::uint8_t body_type = fan::physics::body_type_e::static_body;
          mass_data_t mass_data;
          fan::physics::shape_properties_t shape_properties;

          operator fan::graphics::sprite_properties_t() const;
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
          std::uint8_t draw_mode = fan::graphics::primitive_topology_t::triangle_fan;
          operator fan::graphics::polygon_properties_t() const;
          std::uint8_t body_type = fan::physics::body_type_e::static_body;
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
          std::uint8_t draw_mode = fan::graphics::primitive_topology_t::triangle_strip;
          operator fan::graphics::polygon_properties_t() const;
          std::uint8_t body_type = fan::physics::body_type_e::static_body;
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
        std::uint8_t body_type = fan::physics::body_type_e::static_body,
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
        std::uint8_t body_type = fan::physics::body_type_e::static_body,
        std::array<fan::physics::shape_properties_t, 4> shape_properties = { {
          {.friction = 0  },  // top
          {.friction = 0.6},  // bottom
          {.friction = 0  },  // left
          {.friction = 0  }   // right
        } }
      );

      struct character2d_t;

      inline constexpr f32_t default_jump_height = 32.f;
      inline constexpr f32_t default_knockback = 20.f;

      struct jump_state_t {
        std::function<void(int jump_state)> on_jump = [](int) {};
        f32_t last_ground_time = 0.f;
        f32_t coyote_time = 0.1f;
        f32_t impulse = default_jump_height;
        bool prev_jump_button = false;
        bool jumping = false;
        bool consumed = false;
        bool double_jump_consumed = false;
        bool on_air_after_jump = false;
        bool handle_jump = true;
        bool allow_double_jump = false;

        void reset();
        bool can_coyote_jump(f32_t current_time) const;
      };
      struct wall_jump_t {
        fan::vec2 normal;
        f32_t slide_speed = 200.f;
        f32_t push_away_force = 1.f;
        bool consumed = false;
      };
      struct movement_e {
        enum {
          side_view,
          top_view
        };
      };
      struct movement_state_t {
        fan::vec2 calculate_velocity(
          const fan::vec2& current_velocity,
          const fan::vec2& direction,
          f32_t dt
        ) const;
        void move_to_direction_raw(fan::physics::body_id_t body, const fan::vec2& direction);
        void move_to_direction(fan::physics::body_id_t body, const fan::vec2& direction);
        void update_ai_orientation(character2d_t& c, const fan::vec2& target_distance);
        void perform_jump(fan::physics::body_id_t body_id, bool jump_condition, fan::vec2* wall_jump_normal = nullptr, wall_jump_t* wall_jump = nullptr);

        f32_t acceleration_force = 120.f;
        f32_t deceleration_force = 300.f;
        f32_t max_speed = 300.f;
        std::uint8_t type = movement_e::side_view;
        jump_state_t jump_state;
        fan::vec2 last_direction = 0;
        fan::vec2 desired_facing = {1, 0};
        fan::vec2 knockback_initial_velocity = 0;
        int knockback_ticks_remaining = 0;
        f32_t knockback_duration = 0.1f;
        bool is_in_knockback = false;
        bool ignore_input = false;
        bool enabled = false;
        bool check_gui = true;
        bool is_wall_sliding = false;
      };

      struct attack_state_t {
        bool can_attack(const fan::vec2& target_distance);
        bool try_attack(character2d_t* character);
        // this is for ai
        bool try_attack(character2d_t* character, const fan::vec2& target_distance);
        void end_attack();

        void take_hit(character2d_t* source, const fan::vec2& hit_direction, f32_t knockback_multiplier = 1.0f);

        f32_t max_health = 50;
        f32_t health = max_health;

        f32_t damage = 10.f;
        f32_t knockback_force = 50.f;
        fan::vec2 attack_range = {100, 50};
        bool is_attacking = false;

        fan::time::timer cooldown_timer {0.5e9, true};
        bool attack_requires_facing_target = true;
        bool took_damage = false;
        bool stun = true;
        character2d_t* target = nullptr; // TODOOO REMOVE
        bool attack_used = false;

        std::function<void()> on_attack_start;
        std::function<void()> on_attack_end;
        std::function<void()> on_death;
      };

      struct ai_behavior_t;
      struct navigation_helper_t {
        bool detect_and_handle_obstacles(character2d_t* character, const ai_behavior_t& ai_behavior, const fan::vec2& direction, fan::vec2 tile_size);
        void add_obstacle(std::function<bool(fan::vec2)> cb);

        bool auto_jump_obstacles = true;
        f32_t jump_lookahead_tiles = 1.0f;
        f32_t stuck_threshold = 0.5f;
        fan::time::timer stuck_timer{0.1e9, true};
        fan::time::timer wall_hit_timer {0.3e9, true};
        int wiggle_dir = 1;
        f32_t prev_x = 0;
        bool was_jumping = false;
        bool is_stuck_state = false;
        std::function<bool(const fan::vec2& position)> on_check_obstacle = [](const fan::vec2&) { return false; };
        std::vector<std::function<bool(fan::vec2)>> obstacle_sources;
      };
      
      struct ai_behavior_t {
        using movement_callback_handle_t = fan::physics::step_callback_nr_t;
        enum behavior_type_e {
          none,
          follow_target,
          flee_from_target,
          patrol
        };

        fan::vec2 get_target_distance(const fan::vec2& current_position) const;
        bool should_move(const fan::vec2& distance) const;

        void enable_ai_follow(character2d_t* target, const fan::vec2& trigger_distance, const fan::vec2& closeup_distance);
        void enable_ai_flee(character2d_t* target, const fan::vec2& trigger_distance, const fan::vec2& closeup_distance);
        void enable_ai_patrol(const std::vector<fan::vec2>& points);
        void update_ai(character2d_t* character, navigation_helper_t& navigation, const fan::vec2& target_position, fan::vec2 tile_size);

        behavior_type_e type = none;
        character2d_t* target = nullptr;
        fan::vec2 trigger_distance = {300, 50};
        fan::vec2 closeup_distance = {100, 50};
        bool auto_flip_to_target = true;
        bool auto_jump_obstacles = true;
        f32_t obstacle_lookahead = 1.5f;
        std::vector<fan::vec2> patrol_points;
        std::size_t current_patrol_index = 0;
        wall_jump_t wall_jump;
      };

    #if defined(FAN_JSON)
      struct character_config_t {
        std::string json_path;
        f32_t aabb_scale = 1.0f;
        fan::vec2 draw_offset{0, 0};
        physics::character2d_t* target = nullptr;
        bool auto_animations = true;
        std::function<bool(character2d_t&)> attack_cb;
        fan::physics::shape_properties_t physics_properties = {.fixed_rotation = true};
      };
    #endif

      struct character2d_t : physics::base_shape_t {
        using physics::base_shape_t::base_shape_t;
        using movement_callback_handle_t = fan::physics::step_callback_nr_t;

        character2d_t();
        template <typename T>
        requires(
          std::is_convertible_v<T, base_shape_t> &&
          !std::is_same_v<std::remove_cvref_t<T>, character2d_t>
            )
          explicit character2d_t(T&& shape) : base_shape_t(std::forward<T>(shape)) {}
        character2d_t(const character2d_t& o);
        character2d_t(character2d_t&& o) noexcept;
        character2d_t& operator=(const character2d_t& o);
        character2d_t& operator=(character2d_t&& o) noexcept;
        ~character2d_t();

        fan::vec3 get_center() const;

        void set_physics_position(const fan::vec2& p);
        void set_shape(fan::graphics::shape_t&& shape);
        void set_physics_body(fan::physics::entity_t&& entity);

        movement_callback_handle_t add_movement_callback(std::function<void()> fn);
        void enable_default_movement(std::uint8_t movement = movement_e::side_view);
        void enable_default_movement(f32_t max_speed, f32_t jump_height, std::uint8_t movement = movement_e::side_view);

        #if defined(FAN_JSON)
        void setup_default_animations(const character_config_t& config);
        #endif

        void process_keyboard_movement(std::uint8_t movement = movement_e::side_view, f32_t friction = 12.f);
        bool is_on_ground() const;
        void request_drop_through();

        f32_t get_max_health() const;
        f32_t get_health() const;
        void set_max_health(f32_t v);
        void set_health(f32_t v);
        void instant_kill();
        bool is_dead() const;
        void reset_health(f32_t max_health = -1.f);

        f32_t get_jump_height() const;
        void set_jump_height(f32_t v);
        void enable_double_jump();

        void set_movement_speed(f32_t max_speed);

        void setup_attack_properties(attack_state_t&& attack_state);
        void take_knockback(character2d_t* source, const fan::vec2& hit_direction, f32_t knockback_multiplier = 1.0f);
        void take_hit(character2d_t* source, const fan::vec2& hit_direction, f32_t knockback_multiplier = 1.0f);
        void take_hit(character2d_t* source);

        void update_animations();
        void cancel_animation();

        bool raycast(
          const character2d_t& target
        );

        void enable_oneway_platforms();

        void squish(fan::vec2 from_size, fan::vec2 to_size, f32_t duration, std::function<f32_t(f32_t)> e = fan::tween::easing::out_elastic);

        fan::graphics::sprite_sheet_controller_t anim_controller;
        attack_state_t attack_state;
        
        movement_state_t movement_state;
        wall_jump_t wall_jump;
        movement_callback_handle_t movement_cb_handle;
        fan::tween::tween_manager_t tweens;
        fan::physics::body_id_t feet[2];
        int combat_frame = 0;
        bool drop_through_requested = false;
        fan::time::timer drop_through_timer;
        bool oneway_enabled = false;
        bool was_on_ground = true;
      };

#if defined(FAN_JSON)
      character2d_t from_json(const character_config_t& config, const std::source_location& callers_path = std::source_location::current());
#endif

      struct attack_hitbox_t {
        struct hitbox_spawn_t {
          int frame = 4;
          std::function<fan::physics::entity_t(const fan::vec2& center, f32_t direction)> create_hitbox;
        };
        struct hitbox_config_t {
          std::vector<hitbox_spawn_t> spawns = {{}};
          std::string attack_animation = "attack0";
          bool track_hit_targets = false;
        };
        struct hitbox_instance_t {
          fan::physics::entity_t hitbox;
          int spawn_frame = -1;
          int destroy_frame = -1;
          bool used = false;
          bool pending_destroy = false;
        };

        void setup(const hitbox_config_t& cfg);
        void update(character2d_t* character);
        void process_destruction();
        void spawn_hitbox(character2d_t* character, int index);
        bool spawned() const;
        bool check_hit(character2d_t* character, int index, character2d_t* target);
        void cleanup(character2d_t* character);
        std::size_t hitbox_count() const;

        hitbox_config_t config;
        std::vector<hitbox_instance_t> instances;
        std::vector<std::uint8_t> hitbox_spawned;
        std::unordered_set<std::uint64_t> hit_enemies;
      };

      struct boss_behavior_t {
        struct config_t {
          f32_t ideal_distance = 400.f;
          f32_t backstep_chance = 0.3f;
          f32_t backstep_duration = 0.6e9;
          f32_t backstep_cooldown = 5.0e9;
          f32_t idle_movement_chance = 0.6f;
          std::pair<std::uint64_t, std::uint64_t> idle_timer_range = {
            static_cast<std::uint64_t>(3.0e9),
            static_cast<std::uint64_t>(6.0e9)
          };
        };

        fan::time::timer idle_timer, backstep_timer, backstep_cooldown;
        bool is_backstepping = false;
        int backstep_dir = 0;

        void restart();
        void init(const config_t& cfg);
        bool update(character2d_t& body, const fan::vec2& target_pos, const config_t& cfg);
      };
      // --------------------------------------------------------------------------------

      struct character_movement_preset_t {

        static void setup_default_controls(fan::graphics::physics::character2d_t& body);
      };
      struct combat_controller_t {
        fan::graphics::physics::attack_hitbox_t hitbox;
        bool did_attack = false;
        f32_t damage = 10.f;

        void setup_attack(fan::graphics::physics::character2d_t& body, const std::string& anim_name, int hit_frame);

        template <typename enemies_t>
        void handle_attack(fan::graphics::physics::character2d_t& body, enemies_t& enemies) {
          for (auto& enemy : enemies) {
            if (!hitbox.check_hit(&body, 0, &enemy.get_body())) continue;
            if (enemy.on_hit(&body, (enemy.get_body().get_position() - body.get_position()).normalize())) {
              break;
            }
          }
          hitbox.update(&body);
        }
      };

      struct ai_character2d_t {
        character2d_t body;
        ai_behavior_t behavior;
        navigation_helper_t navigation;

#if defined(FAN_JSON)
        void open(const character_config_t& properties, fan::vec2 initial_pos, const std::source_location& callers_path = std::source_location::current());
#endif
        void update(fan::vec2 tile_size);
      };

      // -----------------------------------bone stuff-----------------------------------

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
      std::string bone_to_string(int bone);

      struct bone_t {
        fan::graphics::physics::base_shape_t visual;
        fan::physics::joint_id_t joint_id = fan::physics::joint_get_null();
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
          b2BodyId bodyId = fan::physics::body_get_null();
        };

        static bool QueryCallback(b2ShapeId shapeId, void* context);

        mouse_joint_t();
        ~mouse_joint_t();

        fan::physics::body_id_t target_body;
        fan::physics::joint_id_t mouse_joint = fan::physics::joint_get_null();
      };
    } // namespace fan::graphics::physics

    struct trigger_t {
      trigger_t() = default;

      void open(
        auto& trigger_to,
        physics::sprite_t::properties_t p,
        std::function<void(physics::sprite_t&)> on_enter) {

        p.shape_properties.is_sensor = true;
        shape = physics::sprite_t {p};

        shape.on_sensor_enter(trigger_to, [this, on_enter = std::move(on_enter)] {
          on_enter(shape);
        });
      }
      void open(
        auto& trigger_to,
        fan::graphics::sprite_t&& graphics_shape,
        std::function<void(physics::sprite_t&)> on_enter) {

        physics::sprite_t::properties_t sprops;
        sprops.shape_properties.is_sensor = true;
        auto sensor = fan::physics::create_sensor_rectangle(graphics_shape.get_position(), graphics_shape.get_size());
        shape = physics::sprite_t(std::move(graphics_shape), std::move(sensor));
        shape.on_sensor_enter(trigger_to, [this, on_enter = std::move(on_enter)] {
          on_enter(shape);
        });
      }

      trigger_t(const trigger_t&) = delete;
      trigger_t& operator=(const trigger_t&) = delete;

      physics::sprite_t shape;
      bool fired = false;
    };
  } // namespace fan::graphics
} // namespace fan

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
    const fan::vec2& center0 = { 0.f, -32.f },
    const fan::vec2& center1 = { 0.f, 32.f },
    f32_t radius = 16.f,
    const fan::physics::shape_properties_t& shape_properties = {.fixed_rotation = true}
  );
  // creates physics body for visual shape
  fan::physics::entity_t character_capsule(
    const fan::graphics::shape_t& shape,
    f32_t shape_size_multiplier = 1.0,
    const fan::physics::shape_properties_t& physics_properties = { 
      .fixed_rotation = true
    }, std::uint8_t body_type = fan::physics::body_type_e::dynamic_body);
  fan::graphics::physics::character2d_t character_capsule(
    const fan::graphics::physics::capsule_t::properties_t& visual_properties,
    const fan::physics::shape_properties_t& physics_properties = {.fixed_rotation = true}
  );
  fan::graphics::physics::character2d_t character_capsule_sprite(
    const fan::vec3& position,
    const fan::vec2& center0 = { 0.f, -32.f },
    const fan::vec2& center1 = { 0.f, 32.f },
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

export namespace fan::graphics {
  void camera_look_at(fan::graphics::camera_nr_t nr, const fan::graphics::physics::character2d_t& target, f32_t move_speed);
  void camera_look_at(const fan::graphics::physics::character2d_t& target, f32_t move_speed);
}

// dynamic object helpers
export namespace fan::graphics::physics {

  struct elevator_t {
    fan::graphics::sprite_t visual;
    fan::physics::entity_t trigger_sensor;
    fan::physics::entity_t walls[4];

    fan::vec2 start_position;
    fan::vec2 end_position;

    f32_t duration;
    f32_t t;
    bool is_active;
    bool going_up;
    bool walls_created;
    bool waiting_for_player_exit;

    fan::physics::step_callback_nr_t step_cb;

    std::function<void()> on_start_cb = [] {};
    std::function<void()> on_end_cb;

    void init(const fan::graphics::sprite_t& sprite, const fan::vec2& start_pos, const fan::vec2& end_pos, f32_t dur);
    void create_trigger_sensor();
    void create_elevator_box();
    void start();
    void physics_step();
    void sync_visual();
    void update(const fan::physics::entity_t& sensor_triggerer);
    void destroy();
  };

  template <typename TargetTag, typename Registry>
  void proximity_damage(Registry& registry, vec2 pos, f32_t radius, int& hp_pool,
    int dmg, fan::color col, int particles) {
    fan::physics::proximity_trigger<fan::ecs::c_pos, TargetTag>(registry, pos, radius, [&](std::uint32_t e, fan::ecs::c_pos& p) {
      hp_pool -= dmg;
      registry.destroy(e);
      fan::graphics::emit_radial(registry, p.v, col, particles, {50.f,200.f}, {0.2f,0.6f});
    });
  }
}

export namespace fan::physics {
  struct steer_params_t {
    f32_t speed     = 1.f;
    f32_t drag      = 0.9f;
    f32_t sep_scale = 0.5f;
    f32_t arrive_r2 = 1600.f; // squared
  };

  fan::vec2 steer_toward(vec2 pos, vec2 vel, vec2 target, vec2 sep, const steer_params_t& p, f32_t speed_mul = 1.f);

  template <typename WallTag, typename ObstacleTag, typename Registry, typename World, typename OnDamage>
  void push_out_walls(Registry& registry, World& world, vec2& pos, f32_t grid, f32_t dt, bool bash, OnDamage&& on_damage) {
    world.query_radius(pos, grid * 0.75f, [&](std::uint32_t id) {
      vec2 ext = vec2(grid / 2.f - 0.1f);
      if (registry.template has<WallTag>(id) &&
          fan::physics::aabb_t::from_center(registry.template get<fan::ecs::c_pos>(id).v, ext).push_out(pos, 200.f * dt))
        on_damage(id, bash);
      else if (registry.template has<ObstacleTag>(id))
        fan::physics::aabb_t::from_center(registry.template get<fan::ecs::c_pos>(id).v, ext).push_out(pos, 200.f * dt);
    });
  }

  template <typename TargetTag, typename... BlockTags, typename Registry, typename World>
  void tick_bullets(Registry& registry, World& world, f32_t radius, int dmg) {
    registry.template destroy_if<fan::ecs::c_pos, fan::ecs::tag_bullet>([&](fan::ecs::c_pos& p, fan::ecs::tag_bullet&) {
      bool hit = false;
      world.query_radius(p.v, radius, [&](std::uint32_t id) {
        if (hit) return;
        if (registry.template has<TargetTag>(id)) {
          registry.template get<fan::ecs::c_hp>(id).current -= dmg; hit = true;
        } else if ((registry.template has<BlockTags>(id) || ...)) {
          hit = true;
        }
      });
      return hit;
    });
  }

  template <typename Registry, typename World>
  void destroy_bullets_vs_tiles(Registry& registry, World& world, f32_t radius) {
    registry.template destroy_if<fan::ecs::c_pos, fan::ecs::tag_bullet>([&](fan::ecs::c_pos& p, fan::ecs::tag_bullet&) {
      bool hit = false;
      world.query_radius(p.v, radius, [&](std::uint32_t) { hit = true; });
      return hit;
    });
  }

  template <typename... Tags_t, typename Registry_t, typename World_t>
  bool has_los(Registry_t& registry, World_t& world, fan::vec2 src, fan::vec2 tgt) {
    bool hit = false;
    world.raycast(src, tgt, [&](std::uint32_t id) {
      if ((registry.template has<Tags_t>(id) || ...)) {
        hit = true;
      }
    });
    return !hit;
  }

  template <typename Tag_t, typename Registry_t, typename World_t>
  fan::vec2 separation_force(Registry_t& registry, World_t& world, std::uint32_t entity_id, fan::vec2 pos, f32_t radius) {
    return world.separation_force(entity_id, pos, radius, [&](std::uint32_t id) {
      return registry.template has<Tag_t>(id) ? registry.template get<fan::ecs::c_pos>(id).v : pos;
    });
  }
}

#endif

#endif

#endif