#pragma once

#if defined(fan_build_pch)
#if defined(loco_physics)

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

constexpr static f32_t bcol_step_time = 0.001;
#define BCOL_set_Dimension 2
#define BCOL_set_IncludePath FAN_INCLUDE_PATH/fan
#define BCOL_set_prefix bcol
#define BCOL_set_DynamicDeltaFunction 
#define BCOL_set_StoreExtraDataInsideObject 1
#define BCOL_set_ExtraDataInsideObject \
  bcol_t::ShapeID_t shape_id;\
  fan::collider::types_e collider_type; \
  loco_t::shape_t* shape = nullptr;
#include _FAN_PATH(ETC/BCOL/BCOL.h)

namespace fan {
  namespace graphics {
    inline bcol_t bcol;
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
            auto d = gloco->get_delta_time();
            bcol_delta += d;
          }

          if (bcol_delta > bcol_delta_max) {
            bcol_delta = bcol_delta_max;
          }

          while (bcol_delta >= bcol_step_time) {
            fan::graphics::bcol.Step(bcol_step_time);

            bcol_delta -= bcol_step_time;
          }
        }
      };
    }

    struct collider_static_t : loco_t::shape_t {
      collider_static_t() = default;
      collider_static_t(const loco_t::shape_t& shape)
        : loco_t::shape_t(shape){
        bcol_t::ObjectProperties_t p;
        p.Position = get_position();
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
        bcol.GetObject_Position(oid);
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
      collider_dynamic_t(const loco_t::shape_t& shape)
        : loco_t::shape_t(shape) {
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
      }
      void close() {
        bcol.UnlinkObject(oid);
        bcol.RecycleObject(oid);
      }
      fan::vec2 get_collider_position() const {
        return bcol.GetObject_Position(oid);
      }

      void set_velocity(const fan::vec2& v) {
        bcol.SetObject_Velocity(oid, v);
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
      }
      void close() {
        bcol.UnlinkObject(oid);
        bcol.RecycleObject(oid);
      }
      fan::vec2 get_collider_position() const {
        return bcol.GetObject_Position(oid);
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
      collider_sensor_t(const fan::vec2& position, const fan::vec2& size) {
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
#endif