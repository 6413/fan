#define car_ ((OFFSETLESS(this, pile_t, network_client))->car)

struct network_client_t {

  network_client_t() :
    client(
      "127.0.0.1", 7777,
      // send callback
      [this]() {
        fan::json shape_data = *dynamic_cast<fan::graphics::shape_t*>(&car_.body);
        shape_data["linear_velocity"] = car_.body.get_linear_velocity();
        shape_data["angular_velocity"] = car_.body.get_angular_velocity();
        return shape_data.dump();
      },
      // receive callback
      [this](const std::vector<fan::graphics::shape_t>& shapes, const std::string& payload) {
        try {
          if (!car) {
            car = new car_t;
            car->is_local = false;
            car->open(shapes[0].get_position());
          }

          peers = shapes;
          peers[0].set_size(0);

          // hardcoded to one car
          fan::json extra_shape_data = fan::json::parse(payload);

          fan::vec2 server_pos = peers[0].get_position();
          fan::vec2 server_vel = extra_shape_data["linear_velocity"];
          f32_t server_angle = peers[0].get_angle().z;
          f32_t server_ang_vel = extra_shape_data["angular_velocity"];

          fan::vec2 local_pos = car->body.get_position();
          fan::vec2 local_vel = car->body.get_linear_velocity();
          f32_t local_angle = car->body.get_angle().z;
          f32_t local_ang_vel = car->body.get_angular_velocity();

          //f32_t dt = fan::physics::physics_timestep;

          fan::vec2 pos_delta = server_pos - local_pos;
          fan::vec2 instant_vel = pos_delta * 10.f;

          f32_t angle_delta = server_angle - local_angle;
          f32_t instant_ang_vel = angle_delta * 10.f;
          car->body.set_linear_velocity(instant_vel);
          car->body.set_angular_velocity(fan::physics::physics_to_render(fan::vec2(instant_ang_vel)).x);
        }
        catch (...) {

        }
      }
    )
  {
    
  }

  void step() {
    if (car) {
      car->step();
    }
  }

  fan::event::task_t global_task;
  fan::event::task_t write_task;
  fan::event::task_t listen_task;

  bool connected_to_server = false;
  //                                    buffer type
  fan::graphics::network::game_client_t<std::string, fan::graphics::shape_t> client;
  std::vector<fan::graphics::shape_t> peers;
  car_t* car = 0;
}network_client;
#undef player_