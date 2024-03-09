loco_t::shape_t* shape;
fan::graphics::circle_t r;
void open(void* sod) {
  auto pile = OFFSETLESS((loco_t*)gloco, pile_t, loco);

  shape = &pile->stage_loader->get_id(this, "unlit_sprite_tire");
  fan::vec2 size = shape->get_size();
  fan::vec2 rp = fan::vec2(0, -size.y);
  shape->set_rotation_point(rp);
}

void close() {

}

void update() {
  static f32_t x = 0;
  shape->set_size(sin(x) * 100);
  x += gloco->get_delta_time();
  fan::vec2 size = shape->get_size();
  fan::vec2 rp = fan::vec2(0, -size.y);
  shape->set_rotation_point(rp);
  r = fan::graphics::circle_t{{
      .position = fan::vec3(fan::vec2(shape->get_position()) - rp, 0),
      .radius = 10,
      .color = fan::colors::red,
      .blending = true
    }};
  shape->set_angle(shape->get_angle() + gloco->get_delta_time() * 10);
}
