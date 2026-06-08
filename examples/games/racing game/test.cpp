import fan;

int main() {

  fan::graphics::engine_t engine;
  fan::graphics::physics::debug_draw(true);
  engine.physics_context.set_gravity(0);
  fan::graphics::interactive_camera_t ic;
  ic.pan_with_middle_mouse = true;
  fan::graphics::physics::rectangle_t r{ {
      .shape_properties{.is_sensor=true}
  } };

  fan::graphics::physics::rectangle_t c{ {
    .position=500,
    .color=fan::colors::red,
    .body_type=fan::physics::body_type_e::dynamic_body
  } };

  fan::time::timer timer;
  timer.start();

  /*
   engine.physics_context.sensor_events.begin_touch_event_cb = [&timer] (auto& ev) {
    fan::graphics::gui::printf("b2SensorBeginTouchEvent:\n");
    fan::graphics::gui::printf("\tTime: {}\n", timer.elapsed() / 1e9);
    fan::graphics::gui::printf("\tsensorShapeId.index1: {}\n", ev.sensorShapeId.index1);
    fan::graphics::gui::printf("\tvisitorShapeId.index1: {}\n", ev.visitorShapeId.index1);
  };

  engine.physics_context.sensor_events.end_touch_event_cb = [&timer] (auto& ev) {
    fan::graphics::gui::printf("b2SensorEndTouchEvent:\n");
    fan::graphics::gui::printf("\tTime: {}\n", timer.elapsed() / 1e9);
    fan::graphics::gui::printf("\tsensorShapeId.index1: {}\n", ev.sensorShapeId.index1);
    fan::graphics::gui::printf("\tvisitorShapeId.index1: {}\n", ev.visitorShapeId.index1);
  };
  */

  engine.physics_context.sensor_events.begin_touch_event_cb = [&timer] (auto& ev) {
    fan::graphics::gui::printf("b2SensorBeginTouchEvent:\n    Time:{} \n    sensorShapeId.index1:{} \n    visitorShapeId.index1:{}", timer.elapsed()/1e9, ev.sensorShapeId.index1, ev.visitorShapeId.index1);
  };

  engine.physics_context.sensor_events.end_touch_event_cb = [&timer] (auto& ev) {
    fan::graphics::gui::printf("b2SensorEndTouchEvent:\n    Time:{} \n    sensorShapeId.index1:{} \n    visitorShapeId.index1:{}", timer.elapsed()/1e9, ev.sensorShapeId.index1, ev.visitorShapeId.index1);
  };


  engine.loop([&] {
    //if (fan::physics::is_on_sensor(c, r)) {
    //  fan::print("A");
    //}
     fan::vec2 velocity{0, 0};
    float speed = 300.0f; 
    
    if (fan::window::is_key_down(fan::key_w)) {
      velocity.y -= speed;
    }
    if (fan::window::is_key_down(fan::key_s)) {
      velocity.y += speed;
    }
    if (fan::window::is_key_down(fan::key_a)) {
      velocity.x -= speed;
    }
    if (fan::window::is_key_down(fan::key_d)) {
      velocity.x += speed;
    }
    
    c.set_linear_velocity(velocity);

    engine.physics_context.step(engine.delta_time);
  });
}