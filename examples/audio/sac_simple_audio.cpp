#include <fan/pch.h>

// argv[1] == audio/w_voice.sac 
int main(int argc, char** argv) {
  fan::graphics::engine_t engine;

  fan::audio::piece_t piece = fan::audio::open_piece("audio/output.sac");
  uint32_t group_id = 0;
  bool loop = true;
  //fan::audio::play(piece, group_id, loop);

  fan::audio::set_volume(0.01);
  f32_t volume = fan::audio::get_volume();

  engine.loop([&] {
    ImGui::Begin("audio controls");
    if (fan::graphics::audio_button("click me")) {

    }
    if (ImGui::Button("toggle pause")) {
      static int audio_toggle = 0;
      ((audio_toggle++)& 1) == 0 ? fan::audio::pause() : fan::audio::resume();

    }
    if (ImGui::DragFloat("volume", &volume, 0.01f, 0.0f, 1.0f)) {
      fan::audio::set_volume(volume);
    }
    ImGui::End();
  });

  return 0;
}