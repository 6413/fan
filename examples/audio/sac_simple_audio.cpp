#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#include <fan/types/types.h>

#ifndef WITCH_INCLUDE_PATH
  #define WITCH_INCLUDE_PATH WITCH
#endif
#include _INCLUDE_TOKEN(WITCH_INCLUDE_PATH,WITCH.h)
#include <fan/audio/audio.h>

#include <fan/pch.h>

// argv[1] == audio/w_voice.sac 
int main(int argc, char** argv) {
  if (argc != 2) {
    fan::print("usage: .exe *.sac");
    return 1;
  }

  loco_t loco;

  fan::system_audio_t system_audio;

  if (system_audio.Open() != 0) {
    fan::throw_error("failed to open fan audio");
  }

  fan::audio_t audio;
  audio.bind(&system_audio);

  fan::audio_t::piece_t piece;
  sint32_t err = audio.Open(&piece, argv[1], 0);
  if (err != 0) {
    fan::throw_error("failed to open piece:", err);
  }

  {
    fan::audio_t::PropertiesSoundPlay_t p;
    p.Flags.Loop = true;
    p.GroupID = 0;
    audio.SoundPlay(&piece, &p);
  }

  audio.SetVolume(0.01);
  f32_t volume = audio.GetVolume();

  loco.loop([&] {
    ImGui::Begin("audio controls");
    if (ImGui::DragFloat("volume", &volume, 0.01f, 0.0f, 1.0f)) {
      audio.SetVolume(volume);
    }
    ImGui::End();
  });

  audio.unbind();

  system_audio.Close();

  return 0;
}