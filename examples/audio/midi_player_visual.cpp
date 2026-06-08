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

#include <fan/imgui/implot.h>

import fan;

#include <fan/graphics/types.h>


#include <fan/audio/midi_parser.h>
#include <fan/audio/midi_player.h>

void load_audio_pieces() {
  for (int i = 1; i <= 88; ++i) {
    std::string path = "audio/steinway keys/" + (std::to_string(i)) + ".sac";
    pieces.push_back(fan::audio::open_piece(path));
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

struct envelope_t {

  static constexpr f32_t attack_time = 0.003f;        
  static constexpr f32_t decay_time = 0.9f;           
  static constexpr f32_t sustain_level = 0.35f;       
  static constexpr f32_t release_time = 2.0f;         

  static constexpr uint32_t sample_rate = fan::system_audio_t::_constants::opus_decode_sample_rate;
  static constexpr uint32_t channel_count = fan::system_audio_t::_constants::ChannelAmount;
  static constexpr int attack_samples = attack_time * sample_rate;
  static constexpr int decay_samples = decay_time * sample_rate;
  static constexpr int sustain_samples = sample_rate;
  static constexpr int release_samples = release_time * sample_rate;

  static constexpr int total_samples = channel_count * (attack_samples + decay_samples + sustain_samples + release_samples);

  // hardcoded for 2 channels
  static std::vector<f32_t> generate() {
    std::vector<f32_t> envelope;
    envelope.reserve(total_samples);

    for (int i = 0; i < attack_samples; ++i) {
      f32_t value = f32_t(i) / attack_samples;
      envelope.push_back(value);
      envelope.push_back(value * 0.98);
    }
    for (int i = 0; i < decay_samples; ++i) {
      f32_t value = 1.0f - (1.0f - sustain_level) * (f32_t(i) / decay_samples);
      envelope.push_back(value);
      envelope.push_back(value * 0.99);
    }
     f32_t last_sustain_value = 0.0f;
     for (int i = 0; i < sustain_samples; ++i) {
       //f32_t decay_factor = std::pow(0.01f, static_cast<f32_t>(i) / sustain_samples);
       f32_t decay_factor = 1.0f - (1.0f - sustain_level) * (f32_t(i) / sustain_samples);
       f32_t value = sustain_level * decay_factor;
       last_sustain_value = value; // store last value
       envelope.push_back(value);
       envelope.push_back(value);
     }

     if (last_sustain_value < 1e-5f) {
       for (int i = 0; i < release_samples; ++i) {
         envelope.push_back(0.0f);
         envelope.push_back(0.0f);
       }
     }
     else {
       for (int i = 0; i < release_samples; ++i) {
         f32_t value = last_sustain_value * (1.0f - static_cast<f32_t>(i) / release_samples);
         envelope.push_back(value);
         envelope.push_back(value * 0.97f);
       }
     }
    return envelope;
  }
};

std::vector<f32_t> envelope = envelope_t::generate();

struct key_info_t {
  int position = 0;
  f32_t velocity = 0;
  fan::audio_t::SoundPlayID_t play_id;
};

std::unordered_map<fan::audio_t::_piece_t*, std::vector<key_info_t>> key_info;

void play_event_group(const EventGroup& group) {
  std::vector<int> notes_to_play;
  std::vector<int> notes_to_stop;
  std::vector<uint8_t> note_velocities;

  for (const auto& event : group.events) {
    constexpr int midi_key_offset = 21; // A0

    if (event.status == MIDI_STATUS_NOTE_ON && event.param2 > 0) {
      int index = event.param1 - midi_key_offset;
      if (index >= 0 && index < pieces.size()) {
        notes_to_play.push_back(index);
        note_velocities.push_back(event.param2);
      }
    }
    else if (event.status == MIDI_STATUS_NOTE_OFF ||
      (event.status == MIDI_STATUS_NOTE_ON && event.param2 == 0))
    {
      int index = event.param1 - midi_key_offset;
      notes_to_stop.push_back(index);
    }
  }

  for (size_t i = 0; i < notes_to_play.size(); i++) {
    int index = notes_to_play[i];
    uint8_t velocity = note_velocities[i];
    
    if (index >= 0 && index < pieces.size()) {
      auto found = key_info.find(pieces[index]._piece);
      if (found == key_info.end()) {
        key_info[pieces[index]._piece] = std::vector<key_info_t>();
        found = key_info.find(pieces[index]._piece);
      }

      key_info_t new_key_info;
      new_key_info.position = 0;
      new_key_info.velocity = velocity / 127.0f;
      
      if (found->second.size() + 1 > found->second.capacity()) {
        found->second.reserve(found->second.size() + 10);
      }
      
      found->second.push_back(new_key_info);
      auto& sound = found->second.back();

      fan::color key_color = fan::color::rgb(
        168 + (87 * (1.0f - sound.velocity)),
        59 * sound.velocity,
        59 * sound.velocity
      );
      piano.keys[index].visual.set_color(key_color);

      sound.play_id = fan::audio::play(pieces[index]);
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


void apply_envelope(fan::graphics::engine_t& engine) {
  auto lambda = [](fan::system_audio_t::Process_t* Process, fan::audio_t::_piece_t* piece, uint32_t play_id, f32_t* samples, uint32_t samplesi) {
    const int attack_samples = envelope_t::attack_samples;
    const int decay_samples = envelope_t::decay_samples;
    const int sustain_samples = envelope_t::sustain_samples;
    const int release_samples = envelope_t::release_samples;

    const int pre_release_samples = attack_samples + decay_samples + sustain_samples;

    auto found = key_info.find(piece);
    if (found == key_info.end()) {
      return;
    }

    for (int j = 0; j < found->second.size(); ++j) {
      auto& sound = found->second[j];
      if (sound.play_id.nr.NRI != play_id) {
        continue;
      }

      for (int i = 0; i < samplesi; ++i) {
        int envelope_index = sound.position;
        ++sound.position;

        int left_idx = envelope_index * 2;
        int right_idx = left_idx + 1;

        if (left_idx >= envelope.size() || right_idx >= envelope.size()) {
          sound.position = 0;
          if (sound.play_id.iic() == false) {
            fan::audio::stop(sound.play_id);
            sound.play_id.sic();
          }
          std::memset(samples, 0, samplesi * 2 * sizeof(f32_t));
          
          if (j < found->second.size()) {
            found->second.erase(found->second.begin() + j);
            --j;
          }
          break;
        }
        
        samples[i * 2] *= envelope[left_idx] * sound.velocity;
        samples[i * 2 + 1] *= envelope[right_idx] * sound.velocity;
      }
      
      if (sound.play_id.nr.NRI == play_id) {
        break;
      }
    }
  };
  
  for (auto& piece : pieces) {
    piece._piece->buffer_end_cb = lambda;
    if (key_info.find(piece._piece) == key_info.end()) {
      key_info[piece._piece] = std::vector<key_info_t>();
    }
  }
}

int main() {
  fan::graphics::engine_t engine;
  
  engine.clear_color = fan::colors::gray / 2;

  load_audio_pieces();

  std::string midi_file_path = "audio/rhapsody22.mid";
  midi_player midi_player;

  //auto queue_processor = process_event_queue();

  apply_envelope(engine);

  auto midi_task = [&]()->fan::event::task_t {
    midi_player = co_await create_midi_player(
      midi_file_path, 
      midi_event_cb,
      meta_event_cb,
      sysex_event_cb
    );
    
    while (co_await process_midi_events(midi_player)) {
      co_await fan::co_sleep(1);
    }
  }();

  f32_t volume = fan::audio::get_volume();

  piano.init(engine);

  midi_timer_callback();


  engine.loop([&] {
    midi_timer_callback();

    fan_graphics_gui_window("audio controls") {
      if (fan::graphics::gui::button("play")) {
        start_playback();
      }
      static f32_t playback_speed = 1.0;
      if (fan::graphics::gui::drag("bpm", &playback_speed, 0.01, 0.01, 1.0f)) {
        set_playback_speed(playback_speed);
      }

      if (fan::graphics::gui::drag("volume", &volume, 0.01f, 0.0f, 1.0f)) {
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