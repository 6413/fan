#pragma once

#if defined(fan_physics)

// change get_collider_position to get_position and use get_sprite_position etc

namespace fan {
  namespace collider {
    enum class types_e {
      collider_static,
      collider_dynamic,
      collider_hidden,
      collider_sensor
    };
  }
}

static f32_t bcol_step_time = 0.001f;
#define BCOL_set_Dimension 2
#define BCOL_set_IncludePath 
#define BCOL_set_prefix bcol

namespace fan {
  namespace graphics {
    inline fan::vec2 gravity = 0;
    // constant_friction is buggy
    inline f32_t constant_friction = 0;
    inline f32_t bump_friction = 0;
    static void set_gravity(const fan::vec2& v) {
      gravity = v;
    }
    static void set_constant_friction(f32_t v) {
      constant_friction = v;
    }
    static void set_bump_friction(f32_t v) {
      bump_friction = v;
    }
    static void set_step(f32_t v) {
      bcol_step_time = v;
    }
  }
}

//#define BCOL_set_ConstantFriction fan::graphics::constant_friction
//#define BCOL_set_ConstantBumpFriction fan::graphics::bump_friction
#define BCOL_set_DynamicDeltaFunction ObjectData0->Velocity += fan::graphics::gravity;
#define BCOL_set_StoreExtraDataInsideObject 1
#define BCOL_set_ExtraDataInsideObject \
  bcol_t::ShapeID_t shape_id;\
  fan::collider::types_e collider_type; \
  loco_t::shape_t* shape = nullptr; \
  uint8_t userdata[256];
#include <BCOL/BCOL.h>

namespace fan {
  namespace graphics {
    inline bcol_t bcol;
    inline std::vector<std::function<void()>> bcol_update;
    inline void open_bcol() {
      bcol_t::OpenProperties_t OpenProperties;

      bcol.Open(OpenProperties);
      bcol.PreSolve_Shape_cb =
        [](
          bcol_t* bcol,
          const bcol_t::ShapeInfoPack_t* sip0,
          const bcol_t::ShapeInfoPack_t* sip1,
          bcol_t::Contact_Shape_t* Contact
          ) {

        };
      //bcol.PreSolve_Shape_cb = ...

      auto nr = gloco->m_update_callback.NewNodeLast();
      gloco->m_update_callback[nr] = [] (auto* loco) {
        {
          static f32_t bcol_delta = 0;
          const f32_t bcol_delta_max = 2;

          {
            bcol_delta += gloco->delta_time;
          }

          if (bcol_delta > bcol_delta_max) {
            bcol_delta = bcol_delta_max;
          }

          while (bcol_delta >= bcol_step_time) {
            for (const auto& i : bcol_update) {
              i();
            }
            fan::graphics::bcol.Step(bcol_step_time);
            bcol_delta -= bcol_step_time;
          }
          bcol_update.clear();
        }
      };
    }
    inline void close_bcol() {
      bcol.Close();
    }

    struct collider_static_t : loco_t::shape_t {
      collider_static_t() = default;
      collider_static_t(const loco_t::shape_t& shape)
        : loco_t::shape_t(shape){
        bcol_t::ObjectProperties_t p;
        loco_t::shape_t s = shape;
        p.Position = s.get_position();
        bcol_t::ShapeProperties_Rectangle_t sp;
        sp.Position = 0;
        sp.Size = get_size();
        oid = bcol.NewObject(&p, bcol_t::ObjectFlag::Constant);
        auto shape_id = bcol.NewShape_Rectangle(oid, &sp);
        auto* data = bcol.GetObjectExtraData(oid);
        data->shape = dynamic_cast<loco_t::shape_t*>(this);
        data->shape_id = shape_id;
        data->collider_type = fan::collider::types_e::collider_static;
      }
      fan::vec2 get_position() const {
        return bcol.GetObject_Position(oid);
      }
      void set_position(const fan::vec2& position) {
        bcol.SetObject_Position(oid, position);
      }
      void close() {
        bcol.UnlinkObject(oid);
        bcol.RecycleObject(oid);
      }
      bcol_t::ObjectID_t oid;
    };
    struct collider_dynamic_t : loco_t::shape_t {
      collider_dynamic_t() = default;
      collider_dynamic_t(loco_t::shape_t&& shape)
        : loco_t::shape_t(std::move(shape)) {
        init();
      }
      collider_dynamic_t(const loco_t::shape_t& shape)
        : loco_t::shape_t(shape) {
       init();
      }
      void init() {
        bcol_t::ObjectProperties_t p;
        p.Position = get_position();
        bcol_t::ShapeProperties_Circle_t sp;
        sp.Position = 0;
        sp.Size = get_size().max();
        oid = bcol.NewObject(&p, 0);
        auto shape_id = bcol.NewShape_Circle(oid, &sp);
        auto* data = bcol.GetObjectExtraData(oid);
        data->shape = dynamic_cast<loco_t::shape_t*>(this);
        data->shape_id = shape_id;
        data->collider_type = fan::collider::types_e::collider_dynamic;
        set_velocity(0);
      }
      void close() {
        bcol.UnlinkObject(oid);
        bcol.RecycleObject(oid);
      }
      fan::vec2 get_collider_position() const {
        return bcol.GetObject_Position(oid);
      }

      fan::vec2 get_velocity() {
        return bcol.GetObject_Velocity(oid);
      }
      void set_velocity(const fan::vec2& v) {
        bcol.SetObject_Velocity(oid, v);
      }
      void set_collider_size(const fan::vec2& v) {
        auto* data = bcol.GetObjectExtraData(oid);
        auto SData = bcol.ShapeData_Circle_Get(data->shape_id);
        SData->Size = v.x;
      }

      void move(const fan::vec2& speed) {
        f32_t dt = gloco->delta_time;
        f32_t multiplier = 1;

        fan::vec2 velocity = 0;
        if (gloco->window.key_pressed(fan::key_shift)) {
          multiplier = 3;
        }
        if (gloco->window.key_pressed(fan::key_d)) {
          velocity.x = speed.x * multiplier;
        }
        else if (gloco->window.key_pressed(fan::key_a)) {
          velocity.x = -speed.x * multiplier;
        }
        else {
          velocity.x = 0;
        }

        if (gloco->window.key_pressed(fan::key_w)) {
          velocity.y = -speed.y * multiplier;
        }
        else if (gloco->window.key_pressed(fan::key_s)) {
          velocity.y = speed.y * multiplier;
        }
        else {
          velocity.y = 0;
        }

        set_velocity(get_velocity() + velocity);

        set_position(get_collider_position());
      }

      bcol_t::ObjectID_t oid;
    };
    struct collider_hidden_t {
      collider_hidden_t() = default;
      collider_hidden_t(const fan::vec2& position, const fan::vec2& size) {
        bcol_t::ObjectProperties_t p;
        p.Position = position;
        bcol_t::ShapeProperties_Rectangle_t sp;
        sp.Position = 0;
        sp.Size = size;
        oid = bcol.NewObject(&p, bcol_t::ObjectFlag::Constant);
        auto shape_id = bcol.NewShape_Rectangle(oid, &sp);
        auto* data = bcol.GetObjectExtraData(oid);
        data->shape_id = shape_id;
        data->collider_type = fan::collider::types_e::collider_hidden;
      }
      fan::vec2 get_position() const {
        return bcol.GetObject_Position(oid);
      }
      void set_position(const fan::vec2& position) {
        bcol.SetObject_Position(oid, position);
      }
      void close() {
        bcol.UnlinkObject(oid);
        bcol.RecycleObject(oid);
      }
      bcol_t::ObjectID_t oid;
    };
    struct collider_dynamic_hidden_t {
      collider_dynamic_hidden_t() = default;
      collider_dynamic_hidden_t(const fan::vec2& position, const fan::vec2& size) {
        bcol_t::ObjectProperties_t p;
        p.Position = position;
        bcol_t::ShapeProperties_Circle_t sp;
        sp.Position = 0;
        sp.Size = size.x;
        oid = bcol.NewObject(&p, 0);
        auto shape_id = bcol.NewShape_Circle(oid, &sp);
        auto* data = bcol.GetObjectExtraData(oid);
        data->shape_id = shape_id;
        data->collider_type = fan::collider::types_e::collider_dynamic;
        set_velocity(0);
      }
      void close() {
        bcol.UnlinkObject(oid);
        bcol.RecycleObject(oid);
      }
      fan::vec2 get_collider_position() const {
        return bcol.GetObject_Position(oid);
      }

      fan::vec2 get_velocity() {
        return bcol.GetObject_Velocity(oid);
      }
      void set_velocity(const fan::vec2& v) {
        bcol.SetObject_Velocity(oid, v);
      }
      void set_position(const fan::vec2& position) {
        bcol.SetObject_Position(oid, position);
      }
      bcol_t::ObjectID_t oid;
    };
    struct collider_sensor_t {
      collider_sensor_t() = default;
      template <typename T = int>
      collider_sensor_t(const fan::vec2& position, const fan::vec2& size, T userdata = 0) {
        bcol_t::ObjectProperties_t p;
        p.Position = position;
        bcol_t::ShapeProperties_Rectangle_t sp;
        sp.Position = 0;
        sp.Size = size;
        oid = bcol.NewObject(&p, bcol_t::ObjectFlag::Constant);
        auto shape_id = bcol.NewShape_Rectangle(oid, &sp);
        auto* data = bcol.GetObjectExtraData(oid);
        data->shape_id = shape_id;
        data->collider_type = fan::collider::types_e::collider_sensor;
        static_assert(sizeof(T) <= sizeof(data->userdata), "too big userdata");
        std::memcpy(data->userdata, &userdata, sizeof(T));
      }
      void close() {
        bcol.UnlinkObject(oid);
        bcol.RecycleObject(oid);
      }
      fan::vec2 get_position() const {
        return bcol.GetObject_Position(oid);
      }
      void set_position(const fan::vec2& position) {
        bcol.SetObject_Position(oid, position);
      }
      bcol_t::ObjectID_t oid;
    };
  }
}

#endif