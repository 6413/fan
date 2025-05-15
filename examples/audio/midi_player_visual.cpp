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

void load_audio_pieces() {
  static constexpr const char* notes[] = {
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
}


struct piano_key_t {
  fan::graphics::rectangle_t visual;
  bool is_pressed = false;
};

static constexpr f32_t key_pad = 4.f;

static constexpr int first_midi_note = 21; // A0
static constexpr int last_midi_note = 108; // C8
static constexpr int total_keys = last_midi_note - first_midi_note + 1;

const std::array<std::string, 12> note_names = {
  "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
};


struct piano_t {

  void init(fan::graphics::engine_t& engine) {

    f32_t window_width = engine.window.get_size().x;
    f32_t keyboard_position_y = engine.window.get_size().y / 1.3;

    const int total_white_keys = 52;
    fan::vec2 white_key_size;
    white_key_size.x = (window_width / 52) / 2.5;
    white_key_size.y = 150;

    fan::vec2 black_key_size;
    black_key_size.x = white_key_size.x / 1.5;
    black_key_size.y = white_key_size.y / 2;

    float white_key_spacing = white_key_size.x * 2 + key_pad;
    float x_offset = (window_width - (white_key_spacing * total_white_keys)) / 2;

    bool has_black_key_after_white_from_A[7] = {
      true,   // A#
      false,  // 
      true,   // C#
      true,   // D#
      false,  // 
      true,   // F#
      true    // G#
    };

    for (int white_index = 0; white_index < total_white_keys; ++white_index) {
      int pos_in_octave = white_index % 7;

      float white_key_x = white_key_spacing * white_index + x_offset;
      piano_key_t white_key;
      white_key.visual = fan::graphics::rectangle_t{ {
        .position = fan::vec3(fan::vec2(white_key_x, keyboard_position_y), 0),
        .size = white_key_size,
        .color = fan::colors::white
      } };
      keys.push_back(white_key);

      if (has_black_key_after_white_from_A[pos_in_octave] && (white_index + 1 < total_white_keys)) {
        float next_white_key_x = white_key_spacing * (white_index + 1) + x_offset;
        float black_key_x = (white_key_x + next_white_key_x) / 2;

        piano_key_t black_key;
        black_key.visual = fan::graphics::rectangle_t{ {
          .position = fan::vec3(
            fan::vec2(black_key_x, keyboard_position_y - (white_key_size.y - black_key_size.y)),
            1),
          .size = black_key_size,
          .color = fan::colors::black
        } };
        keys.push_back(black_key);
      }
    }
  }

  std::vector<piano_key_t> keys;
}piano;


void play_event_group(const EventGroup& group) {
  fan::print("Playing event group at time:", group.timestamp_ms, "ms with", group.events.size(), "events");

  std::vector<int> notes_to_play;
  std::vector<int> notes_to_stop;

  for (const auto& event : group.events) {
    constexpr int midi_key_offset = 21; // A0

    if (event.status == MIDI_STATUS_NOTE_ON && event.param2 > 0) {
      int index = event.param1 - midi_key_offset;
      if (index >= 0 && index < pieces.size()) {
        notes_to_play.push_back(index);
        //fan::print("  Note ON:", (int)event.param1, "velocity:", (int)event.param2,
        //  "channel:", (int)event.channel);
      }
    }
    else if (event.status == MIDI_STATUS_NOTE_OFF ||
      (event.status == MIDI_STATUS_NOTE_ON && event.param2 == 0))
    {
      int index = event.param1 - midi_key_offset;
      notes_to_stop.push_back(index);
      //  fan::print("  Note OFF:", (int)event.param1, "channel:", (int)event.channel);
    }
  }

  for (int index : notes_to_play) {
    if (index >= 0 && index < pieces.size()) {
      fan::audio::play(pieces[index]);
      piano.keys[index].visual.set_color(fan::color::rgb(168, 59, 59));
    }
  }
  for (int index : notes_to_stop) {
    if (index >= 0 && index < pieces.size()) {
       int note_in_octave = index % 12;

      bool is_black = (
        note_in_octave == 1  ||  // A#
        note_in_octave == 4  ||  // C#
        note_in_octave == 6  ||  // D#
        note_in_octave == 9  ||  // F#
        note_in_octave == 11     // G#
      );
      if (is_black) {
        piano.keys[index].visual.set_color(0);
      }
      else {
        piano.keys[index].visual.set_color(1);
      }
    }
  }
  
}

int main() {
  fan::graphics::engine_t engine;
  
  engine.clear_color = fan::colors::gray / 2;

  load_audio_pieces();

  std::string midi_file_path = "audio/ballade4.mid";
  midi_player midi_player;

  auto queue_processor = process_event_queue();

  auto midi_task = [&]()->fan::event::task_t {
    midi_player = co_await create_midi_player(
      midi_file_path, 
      midi_event_cb,
      meta_event_cb,
      sysex_event_cb
    );
    
    while (co_await process_midi_events(midi_player)) {
      co_await fan::co_sleep(0);
    }
  }();

  f32_t volume = fan::audio::get_volume();

  piano.init(engine);

  engine.loop([&] {
   /* fan_graphics_gui_window("audio controls") {
      
      if (fan::graphics::gui::drag_float("volume", &volume, 0.01f, 0.0f, 1.0f)) {
        fan::audio::set_volume(volume);
      }
    }*/
    

    if (ImPlot::BeginPlot("pcm")) {
      ImPlot::PlotLine("pcm0", engine.system_audio.Out.frames[0], fan::system_audio_t::_constants::CallFrameCount * fan::system_audio_t::_constants::ChannelAmount);
      ImPlot::PlotLine("pcm1", engine.system_audio.Out.frames[1], fan::system_audio_t::_constants::CallFrameCount * fan::system_audio_t::_constants::ChannelAmount);
      ImPlot::EndPlot();
    }
  });
  
  return 0;
}