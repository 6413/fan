module;

#if defined(fan_physics)
  #include <fan/utility.h>
  #include <box2d/box2d.h>
#endif

#include <functional>
#include <cstdlib>
#include <unordered_set>

export module fan.physics.b2_integration;

#if defined(fan_physics)

export import fan.physics.types;

import fan.types.vector;
import fan.print;
import fan.physics.common_context;

#define BLL_set_SafeNext 1
#define BLL_set_AreWeInsideStruct 0
#define BLL_set_prefix physics_step_callbacks
#include <fan/fan_bll_preset.h>
#define BLL_set_Link 1
#define BLL_set_type_node uint16_t
#define BLL_set_NodeDataType std::function<void()>
#define BLL_set_CPP_CopyAtPointerChange 1
#include <BLL/BLL.h>

export namespace fan::physics {
  struct context_t;

  struct global_physics_t {
    context_t* context = nullptr;
    operator context_t* () {
      return context;
    }

    global_physics_t& operator=(context_t* l) {
      context = l;
      return *this;
    }
    context_t* operator->() {
      return context;
    }

    std::function<fan::vec2()> get_gravity;
    std::function<void(const fan::vec2&)> set_gravity;
  };

  // tiny compile boost xd
#if !defined(fan_compiler_msvc)
  inline
  #endif
    thread_local global_physics_t gphysics;
}

export namespace fan {
  namespace physics {
    struct body_id_t;

    void fill_shape_proxy(b2ShapeProxy& proxy, b2ShapeId shape_id, body_id_t body_id);
    bool test_overlap(body_id_t body_a, body_id_t body_b);
    void on_overlap(body_id_t body_a, body_id_t body_b, std::function<void()> callback);

    struct shapes_e {
      enum {
        capsule,
        polygon,
        circle,
        box,
      };
    };

    struct capsule_t : b2Capsule {
      using b2Capsule::b2Capsule;
      capsule_t() = default;
      capsule_t(const b2Capsule& capsule);
    };
    struct polygon_t : b2Polygon {
      using b2Polygon::b2Polygon;
      polygon_t() = default;
      polygon_t(const b2Polygon& polygon);
    };
    struct circle_t : b2Circle {
      using b2Circle::b2Circle;
      circle_t() = default;
      circle_t(const b2Circle& circle);
    };
    struct segment_t : b2Segment {
      using b2Segment::b2Segment;
      segment_t() = default;
      segment_t(const b2Segment& segment);
    };

    struct ray_result_t : b2RayResult {
      b2ShapeId shapeId;
      fan::vec2 point;
      fan::vec2 normal;
      f32_t fraction;
      bool hit;
      operator bool();
    };

    using physics_step_callback_nr_t = physics_step_callbacks_NodeReference_t;

    struct overlap_test_context_t {
      bool found_overlap = false;
      b2ShapeId target_shape = b2_nullShapeId;
    };
    struct overlap_callback_context_t {
      std::function<void()> callback;
      b2ShapeId target_shape = b2_nullShapeId;
    };

    std::vector<std::function<void()>> one_time_commands;

    void queue_one_time_command(std::function<void()> callback);

    struct b2_body_id_hash_t {
      std::size_t operator()(const b2BodyId& id) const;
    };

    struct b2_body_id_equal_t {
      bool operator()(const b2BodyId& a, const b2BodyId& b) const;
    };

    inline constexpr f32_t default_physics_timestep = 1.0 / 244.f;

    struct shape_id_t : b2ShapeId {
      using b2ShapeId::b2ShapeId;
      shape_id_t();
      shape_id_t(const b2ShapeId& shape_id);

      void set_friction(f32_t friction);

      bool is_valid() const;
      operator bool() const;
    };

    struct body_id_t : b2BodyId {
      using b2BodyId::b2BodyId;
      body_id_t();
      body_id_t(const b2BodyId& body_id);

      void set_body(const body_id_t& b);

      bool operator==(const body_id_t& b) const;

      bool operator!=(const body_id_t& b) const;

      operator bool() const;

      operator b2ShapeId() const;

      bool is_valid() const;

      void invalidate();

      void destroy();

      void erase();

      fan::vec2 get_linear_velocity() const;

      void set_linear_velocity(const fan::vec2& v);

      f32_t get_angular_velocity() const;

      void set_angular_velocity(f32_t v);

      void apply_force_center(const fan::vec2& v);
      void apply_linear_impulse_center(const fan::vec2& v);
      void zero_linear_impulse_center();
      void apply_angular_impulse(f32_t v);
      fan::vec2 get_physics_position() const;
      fan::vec2 get_position() const;
      void set_physics_position(const fan::vec2& p);
      b2ShapeId get_shape_id() const;
      f32_t get_density() const;
      f32_t get_friction() const;
      f32_t get_mass() const;
      f32_t get_restitution() const;
      fan::physics::aabb_t get_aabb() const;
      fan::vec2 get_aabb_size() const;
      bool test_overlap(const body_id_t& other) const;
      void on_overlap(const body_id_t& other, std::function<void()> callback);
    };

    struct joint_id_t : b2JointId {
      using b2JointId::b2JointId;
      joint_id_t();
      joint_id_t(const b2JointId& body_id);

      void set_joint(const joint_id_t& b);

      bool operator==(const joint_id_t& b) const;

      bool operator!=(const joint_id_t& b) const;

      bool is_valid();

      void invalidate();

      void destroy();

      void revolute_joint_set_motor_speed(f32_t v);
    };

    struct entity_t : body_id_t {
      using body_id_t::body_id_t;
    };

    fan::vec2 check_wall_contact(body_id_t body_id, shape_id_t* colliding_wall = nullptr);

    void apply_wall_slide(body_id_t body_id, const fan::vec2& wall_normal, f32_t slide_speed = 20.0f);

    struct sensor_events_t {
      struct sensor_contact_t {
        fan::physics::body_id_t sensor_id;
        fan::physics::body_id_t object_id;
        bool is_in_contact = 0;
      };

      std::function<void(b2SensorBeginTouchEvent&)> begin_touch_event_cb = [](b2SensorBeginTouchEvent&) {};
      std::function<void(b2SensorEndTouchEvent&)> end_touch_event_cb = [](b2SensorEndTouchEvent&) {};

      void update(b2WorldId world_id);

      void update_contact(b2BodyId sensor_id, b2BodyId object_id, bool is_in_contact);

      void remove_body_contacts(b2BodyId body_id);

      bool is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id) const;

      std::vector<sensor_contact_t> contacts;
    };

    struct context_t {

      operator b2WorldId& ();

      struct properties_t {
        properties_t();;
        fan::vec2 gravity{ 0, 9.8f };
      };

      context_t(const properties_t& properties = properties_t());

      void set_gravity(const fan::vec2& gravity);

      fan::vec2 get_gravity() const;

      void begin_frame(f32_t dt);

      entity_t create_box(const fan::vec2& position, const fan::vec2& size, f32_t angle = 0, uint8_t body_type = body_type_e::static_body, const shape_properties_t& shape_properties = {});

      entity_t create_rectangle(const fan::vec2& position, const fan::vec2& size, f32_t angle = 0, uint8_t body_type = body_type_e::static_body, const shape_properties_t& shape_properties = {});

      entity_t create_circle(const fan::vec2& position, f32_t radius, f32_t angle = 0, uint8_t body_type = body_type_e::static_body, const shape_properties_t& shape_properties = {});

      fan::physics::entity_t create_capsule(const fan::vec2& position, f32_t angle, const fan::physics::capsule_t& info, uint8_t body_type, const shape_properties_t& shape_properties);

      fan::physics::entity_t create_segment(const fan::vec2& position, const std::vector<fan::vec2>& points, uint8_t body_type, const shape_properties_t& shape_properties);

      fan::physics::entity_t create_polygon(
        const fan::vec2& position,
        f32_t radius,
        const fan::vec2* points,
        int count,
        uint8_t body_type,
        const shape_properties_t& shape_properties
      );
      // a, b and c are local offsets from 'position' (center)
      entity_t create_triangle(
        const fan::vec2& position,
        const fan::vec2& a,
        const fan::vec2& b,
        const fan::vec2& c,
        uint8_t body_type,
        const shape_properties_t& shape_properties
      );

      void step(f32_t dt);

      bool is_on_sensor(fan::physics::body_id_t test_id, fan::physics::body_id_t sensor_id) const;

      ray_result_t raycast(const fan::vec2& src_, const fan::vec2& dst_);

      void on_begin_touch(b2ShapeId shape_a, b2ShapeId shape_b);

      void on_end_touch(b2ShapeId shape_a, b2ShapeId shape_b);

      void on_hit(b2ShapeId shape_a, b2ShapeId shape_b, f32_t approach_speed);

      uint64_t get_shape_key(b2ShapeId shape) const;

      void add_collision(b2ShapeId a, b2ShapeId b);

      void remove_collision(b2ShapeId a, b2ShapeId b);

      void process_collision_events();

      bool is_colliding(b2ShapeId a, b2ShapeId b) const;

      fan::physics::entity_t create_sensor_circle(const fan::vec2& position, f32_t radius);

      fan::physics::entity_t create_sensor_rectangle(const fan::vec2& position, const fan::vec2& size);

      b2WorldId world_id;
      sensor_events_t sensor_events;
      f32_t delta_time = 0;

      struct pair_hash_t {
        size_t operator()(const std::pair<uint64_t, uint64_t>& p) const;
      };

      std::unordered_set<std::pair<uint64_t, uint64_t>, pair_hash_t> active_collisions;
      fan::physics::physics_update_cbs_t* physics_updates = nullptr;
      physics_step_callbacks_t physics_step_callbacks;
      std::function<void()> debug_draw_cb = [] {};
    };

    bool presolve_oneway_collision(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, fan::physics::body_id_t character_body);

    fan::physics::body_id_t deep_copy_body(b2WorldId worldId, fan::physics::body_id_t sourceBodyId);

    void set_pre_solve_callback(b2WorldId world_id, b2PreSolveFcn* fcn, void* context);

    bool is_colliding(const b2ShapeId& a, const b2ShapeId& b);

    fan::physics::entity_t create_sensor_circle(const fan::vec2& position, f32_t radius);

    fan::physics::entity_t create_sensor_rectangle(const fan::vec2& position, const fan::vec2& size);

    physics_step_callback_nr_t add_physics_step_callback(std::function<void()> callback);
    void remove_physics_step_callback(physics_step_callback_nr_t nr);
    // for drawing physics shapes
    fan::physics::physics_update_cbs_t::nr_t add_physics_update(const fan::physics::physics_update_data_t& cb_data);
    void remove_physics_update(fan::physics::physics_update_cbs_t::nr_t nr);

    bool overlap_result_callback(b2ShapeId shape_id, void* context);
    bool overlap_callback_fcn(b2ShapeId shape_id, void* context);
    void fill_shape_proxy(b2ShapeProxy& proxy, b2ShapeId shape_id, body_id_t body_id);
    bool is_point_overlapping(const fan::vec2& position, const fan::vec2& point_size = 0.5f);
    bool test_overlap(body_id_t body_a, body_id_t body_b);
    void on_overlap(body_id_t body_a, body_id_t body_b, std::function<void()> callback);
    void queue_one_time_command(std::function<void()> callback);
    fan::vec2 check_wall_contact(body_id_t body_id, shape_id_t* colliding_wall);
    void apply_wall_slide(body_id_t body_id, const fan::vec2& wall_normal, f32_t slide_speed);
    void wall_jump(body_id_t body_id, const fan::vec2& wall_normal, f32_t push_x, f32_t max_up_speed);
    bool overlap_callback_fcn(b2ShapeId shape_id, void* context);
    bool is_colliding(const b2ShapeId& a, const b2ShapeId& b);
    
    entity_t create_sensor_circle(const fan::vec2& position, f32_t radius);
    entity_t create_sensor_rectangle(const fan::vec2& position, const fan::vec2& size);
    physics_step_callback_nr_t add_physics_step_callback(std::function<void()> callback);
    void remove_physics_step_callback(physics_step_callback_nr_t nr);
    bool presolve_oneway_collision(b2ShapeId shapeIdA, b2ShapeId shapeIdB, b2Manifold* manifold, body_id_t character_body);
    body_id_t deep_copy_body(b2WorldId worldId, body_id_t sourceBodyId);
    void set_pre_solve_callback(b2WorldId world_id, b2PreSolveFcn* fcn, void* context);

    physics_update_cbs_t::nr_t add_physics_update(const physics_update_data_t& cb_data);
    void remove_physics_update(physics_update_cbs_t::nr_t nr);
    bool overlap_result_callback(b2ShapeId shape_id, void* context);
  }
}

#endif