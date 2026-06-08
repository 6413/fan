#include <fan/types/dme.h>

#include <string>
#include <ranges>
#include <string_view>

import fan;

fan::graphics::engine_t engine;
fan::graphics::shape_t server_shape;
fan::graphics::physics::character2d_t client_shape = { 
  fan::graphics::physics::capsule_sprite_t{{
    .position = fan::vec3(fan::vec2(200, 200), 0),
    .center0 = {0.f, -24.f},
    .center1 = {0.f, 32.f},
    .radius = 12,
    .blending = true,
    .body_type = fan::physics::body_type_e::dynamic_body,
    .shape_properties{
      .friction = 0.6f,
      .density = 0.1f,
      .fixed_rotation = true,
    },
  }
}};

struct animations_t : __dme_inherit(animations_t, fan::graphics::animation_t) {
  __dme(idle, );
  __dme(walk, );
};
animations_t anims;

fan::graphics::render_view_t server_render_view;

fan::event::task_t tcp_server_test() {
  using namespace fan;
  co_await network::tcp_server_listen({ .port = 7777 }, [](const fan::network::tcp_t& client) -> event::task_t {
    std::string json_data;
    network::message_t data;
    while (data = co_await client.read()) {
      json_data += std::string_view(data.buffer);
      if (!data.done || json_data.empty()) {
        continue;
      }
      try {
        //fan::print(json_data);
        server_shape = json_data;
        server_shape.set_render_view(server_render_view);
      }
      catch (const std::exception& e) {
        fan::print_warning("Failed to deserialize rectangle: " + std::string(e.what()));
      }
      json_data.clear();
    }
  });
}

fan::event::task_t tcp_client_test() {
  using namespace fan;

  while (1) {
    try {
      fan::network::tcp_t client;
      co_await client.connect("127.0.0.1", 7777);

      ssize_t error = network::error_code::ok;
      while (error == network::error_code::ok) {
        error = co_await client.write(client_shape);
        static constexpr f32_t fps = 60.f;
        co_await fan::co_sleep(1000.0 / fps);
      }
    }
    catch (std::exception& e) {
      fan::print_warning(std::string("Client error:") + e.what());
    }
    co_await fan::co_sleep(1000); // retry connection every 1s
  }
}

void split_screen(fan::graphics::line_t& splitter) {
  auto size = engine.window.get_size();
  f32_t mid_x = size.x / 2;

  splitter.set_line({mid_x, 0}, {mid_x, size.y});

  server_render_view.set({0, mid_x}, {0, size.y}, {mid_x, 0}, {mid_x, size.y});
  engine.orthographic_render_view.set({0, mid_x}, {0, size.y}, {0, 0}, {mid_x, size.y});
  static auto walls = fan::graphics::physics::create_stroked_rectangle({mid_x / 2.f, size.y / 2.f}, {mid_x / 2.f, size.y / 2.f}, 25.f);
  static auto walls2 = fan::graphics::physics::create_stroked_rectangle({mid_x / 2.f, size.y / 2.f}, {mid_x / 2.f, size.y / 2.f}, 25.f);
  for (auto& wall : walls2) {
    wall.set_render_view(server_render_view);
  }
  fan::graphics::gui::text("Local view");
  fan::graphics::gui::text_at("Peer view", fan::vec2(10.f + mid_x, 0));
}

void move_shape_based_on_input() {
  client_shape.process_movement(fan::graphics::physics::character2d_t::movement_e::side_view);
  client_shape.update_animation();
}

int main() {
  server_render_view.create();
  try {
    auto tcp_server = tcp_server_test();
    auto tcp_client = tcp_client_test();

    std::string current_directory = "game/";
    // init client animations
    fan::json json_data = fan::graphics::read_json(current_directory + "entities/player/player.json");
    gloco()->sprite_sheets_parse(json_data);
    fan::graphics::map_sprite_sheets(anims);
    client_shape.set_shape(fan::graphics::extract_single_shape(json_data));
    client_shape.set_size(client_shape.get_size() / 2.5);
    client_shape.play_sprite_sheet();


    fan::graphics::line_t screen_splitter;

    engine.loop([&] {
      engine.physics_context.step(engine.delta_time);
      split_screen(screen_splitter);
      move_shape_based_on_input();
    });
  }
  catch (const std::exception& e) {
    fan::print(std::string("Exception:") + e.what());
  }
  server_render_view.remove();
  return 0;
}