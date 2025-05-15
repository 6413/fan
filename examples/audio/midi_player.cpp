#include <winsock2.h>
#undef min
#undef max

#include <string>
#include <coroutine>
#include <cassert>
#include <vector>
#include <stdexcept>
#include <functional>
#include <string.h>
#include <queue>

#include <xaudio2.h>
#include <math.h>
#include <map>
#include <mutex>

#include <fan/types/types.h>
#include <fan/math/math.h>

#include <fan/imgui/implot.h>

import fan;

#include <fan/graphics/types.h>

#include <fan/audio/midi_parser.h>
#include <fan/audio/midi_player.h>
int main() {
  loco_t engine;

  const char* notes[] = {
     "a", "as", "b", "c", "cs", "d", "ds", "e", "f", "fs", "g", "gs"
  };

  pieces.reserve(88);

  int octave_counter = 9;

  //int inital_note_offset = 
  for (int i = 0; i < 8; ++i) {
    for (int note = 0; note < std::size(notes); ++note) {
      if (i == 7 && note == 4) {
        break;
      }
      //fan::print(octave_counter / 12, notes[note]);
      std::string path = "audio/piano keys/" + (std::to_string(octave_counter / 12) + "-" + notes[note]) + ".sac";
      pieces.push_back(fan::audio::open_piece(path));
      octave_counter = (octave_counter + 1);
    }
  }
  std::string midi_file_path = "audio/ballade4.mid";
  midi_player midi_player;

  // Start event queue processor
  auto queue_processor = process_event_queue();

  // MIDI file parser task
  auto midi_task = [&]()->fan::event::task_t {
    midi_player = co_await create_midi_player(
      midi_file_path,
      midi_event_cb,
      meta_event_cb,
      sysex_event_cb
    );

    // Process all MIDI events as fast as possible
    // The timing will be handled by the event queue processor
    while (co_await process_midi_events(midi_player)) {
      // Just yield to allow other tasks to run
      co_await fan::co_sleep(0);
    }
    }();

  f32_t volume = fan::audio::get_volume();

  engine.loop([&] {
    fan_graphics_gui_window("audio controls") {
      if (fan::graphics::gui::drag_float("volume", &volume, 0.01f, 0.0f, 1.0f)) {
        fan::audio::set_volume(volume);
      }
    }

    if (ImPlot::BeginPlot("pcm")) {
      ImPlot::PlotLine("pcm0", engine.system_audio.Out.frames[0], fan::system_audio_t::_constants::CallFrameCount * fan::system_audio_t::_constants::ChannelAmount);
      ImPlot::PlotLine("pcm1", engine.system_audio.Out.frames[1], fan::system_audio_t::_constants::CallFrameCount * fan::system_audio_t::_constants::ChannelAmount);
      ImPlot::EndPlot();
    }
    });

  return 0;
}