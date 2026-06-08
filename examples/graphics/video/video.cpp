import fan;

using namespace fan::graphics;

int main() {
  engine_t engine;

  video::player_t player;
  player.open("video.mkv");

  auto file_drop_handle = engine.window.on_drop([&](const auto& d) {
    player.reopen(d.paths[0]);
  });

  engine.loop([&] {
    player.update();
    if (auto h = gui::window("media player", 0.f)) {
      player.show();
    }
  });

  return 0;
}