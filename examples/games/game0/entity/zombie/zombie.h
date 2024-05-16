#define set_EntityName zombie
EntityStructBegin

static constexpr fan::vec2 movement_speed{ 25, 25 };
static constexpr f32_t animation_speed = 0.3e+9;

struct EntityBehaviourData_t {

}EntityBehaviourData;

struct EntityData_t {
  fan::graphics::collider_dynamic_hidden_t collider;
  // if player gets too close it triggers attack
  fan::graphics::collider_sensor_t detect_collider;
  loco_t::shapes_t::sprite_sheet_t::nr_t sheet_id;
  fan::vec2 direction = 0;
  fan::time::clock direction_change_timer;
};

loco_t::image_t frames[2];

CONCAT(set_EntityName, _t)() {
  static constexpr const char* directions[] = { "down", "down" };
  for (int j = 0; j < 2; ++j) {
    frames[j].load(fan::format("images/greenslime_{}_{}.webp", directions[j], std::to_string(j + 1)));
  }
}
~CONCAT(set_EntityName, _t)() {
  for (int j = 0; j < 2; ++j) {
    frames[j].unload();
  }
}

EntityMakeStatic(cb_force_remove,
  fan::graphics::EntityList_t*, EntityList,
  fan::graphics::EntityID_t, EntityID
) {
  auto EntityData = ged(EntityID);
  gloco->shapes.sprite_sheet.stop(EntityData->sheet_id);
  delete EntityData;
}

EntityMakeStatic(cb_delta,
  fan::graphics::EntityList_t*, EntityList,
  fan::graphics::EntityID_t, EntityID,
  f32_t, delta
) {
  auto& entity_data = *ged(EntityID);

  fan::vec2 vel = entity_data.collider.get_velocity();

  if (entity_data.direction_change_timer.finished()) {
    vel = fan::random::vec2i(-1, 1) * movement_speed;
    entity_data.direction_change_timer.start(fan::time::nanoseconds(fan::random::value_f32(1e+9, 5e+9)));
  }

  entity_data.collider.set_velocity(vel);
  gloco->shapes.sprite_sheet.set_position(entity_data.sheet_id, entity_data.collider.get_collider_position());
}

fan::graphics::EntityBehaviour_t EntityBehaviour = {
  .IdentifyingAs = EntityIdentify_t::set_EntityName,
  .cb_force_remove = cb_force_remove,
  .cb_delta = cb_delta,
  .UserPTR = &EntityBehaviourData
};

void Add(fan::vec2 Position) {
  auto& entity_data = *new EntityData_t;

  fan::graphics::Entity_t Entity;
  Entity.Behaviour = &EntityBehaviour;
  Entity.UserPTR = (void*)&entity_data;
  auto EntityID = entites_ptr->entity_list.Add(&Entity);

  {
    loco_t::shapes_t::sprite_sheet_t::properties_t ssp;
    ssp.images = frames;
    ssp.count = std::size(frames);
    ssp.animation_speed = animation_speed;
    ssp.blending = true;
    static int depth = 4;
    ssp.position = { Position, depth++ };
    ssp.size = 16;
    ssp.blending = true;

    
    entity_data.sheet_id = gloco->shapes.sprite_sheet.push_back(ssp);
    gloco->shapes.sprite_sheet.start(entity_data.sheet_id);

    entity_data.collider = fan::graphics::collider_dynamic_hidden_t(ssp.position, ssp.size);
    entity_data.direction_change_timer.start(fan::time::nanoseconds(fan::random::value_f32(1e+9, 5e+9)));
  }
 /* {
    BCOL_t::ObjectProperties_t op;
    op.Position = Position;
    op.ExtraData.EntityID = EntityID;
    BCOL_t::ShapeProperties_Circle_t sp;
    sp.Position = 0;

    op.ExtraData.Mark = 0;
    EntityData->COID_itself = Stage->bcol.NewObject(&op, BCOL_t::ObjectFlag::Constant);
    sp.Size = Size.x;
    Stage->bcol.NewShape_Circle(EntityData->COID_itself, &sp);

    op.ExtraData.Mark = 1;
    EntityData->COID_sense = Stage->bcol.NewObject(&op, BCOL_t::ObjectFlag::Constant);
    sp.Size = 96;
    Stage->bcol.NewShape_Circle(EntityData->COID_sense, &sp);
  }*/
}

#include "../EntityStructEnd.h"
