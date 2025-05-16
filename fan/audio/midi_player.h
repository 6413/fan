#pragma once

// Structure to hold MIDI playback state
struct midi_player {

  using on_midi_event_cb_t =  std::function<void(midi_player&, const midi_midi_event&)>;
  using on_meta_event_cb_t =  std::function<void(midi_player&, const midi_meta_event&)>;
  using on_sysex_event_cb_t = std::function<void(midi_player&, const midi_sysex_event&)>;

  // File reader
  fan::io::file::async_read_t file;

  // Parser state
  midi_parser parser;

  // Buffer management
  std::vector<uint8_t> buffer;
  size_t buffer_size = 8192; // Default buffer size (can be adjusted)
  size_t buffer_pos = 0;
  bool eof_reached = false;

  // MIDI header info
  midi_header header;

  // Playback state
  int current_track = 0;
  int64_t current_time = 0;
  bool header_parsed = false;

  // Event callbacks
  on_midi_event_cb_t on_midi_event;
  on_meta_event_cb_t on_meta_event;
  on_sysex_event_cb_t on_sysex_event;

  uint32_t tempo_us_per_quarter = 500000; // 120bpm

  int64_t last_event_time_ms = 0;
  int64_t current_time_ms = 0;

  int time_elapsed = 0;
};

static constexpr const char* key_names[] = { "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#" };
static constexpr const char* scales[] = {"Major", "Minor"};

fan::event::task_value_resume_t<midi_player> create_midi_player(
  const std::string& path,
  midi_player::on_midi_event_cb_t midi_callback,
  midi_player::on_meta_event_cb_t meta_callback = nullptr,
  midi_player::on_sysex_event_cb_t sysex_callback = nullptr) {

  midi_player player;

  // Set callbacks
  player.on_midi_event = midi_callback;
  player.on_meta_event = meta_callback;
  player.on_sysex_event = sysex_callback;

  // Initialize buffer
  player.buffer.resize(player.buffer_size);

  // Open file
  co_await player.file.open(path);

  // Initialize parser
  player.parser = {};

  co_return player;
}

fan::event::task_value_resume_t<bool> fill_buffer(midi_player& player) {
  // If we've reached EOF, return false
  if (player.eof_reached) {
    co_return false;
  }

  // If there's still data in the buffer, move it to the beginning
  if (player.buffer_pos > 0 && player.parser.size > 0) {
    memmove(player.buffer.data(),
      player.buffer.data() + player.buffer_pos,
      player.parser.size);
  }

  // Reset buffer position
  player.buffer_pos = 0;

  // Read more data
  size_t space_available = player.buffer.size() - player.parser.size;
  if (space_available < player.buffer_size / 4) {
    // Expand buffer if we're running out of space
    player.buffer.resize(player.buffer.size() + player.buffer_size);
    space_available = player.buffer.size() - player.parser.size;
  }

  // Read chunk into buffer
  std::string chunk = co_await player.file.read();
  if (chunk.empty()) {
    player.eof_reached = true;
    co_return player.parser.size > 0;
  }

  // Copy new data into buffer
  if (chunk.size() <= space_available) {
    memcpy(player.buffer.data() + player.parser.size, chunk.data(), chunk.size());
    player.parser.size += chunk.size();
  }
  else {
    // If chunk is larger than available space, copy what fits
    memcpy(player.buffer.data() + player.parser.size, chunk.data(), space_available);
    player.parser.size += space_available;
    fan::print("Warning: Buffer overflow - some MIDI data may be lost");
  }

  // Update parser input pointer
  player.parser.in = player.buffer.data();

  co_return true;
}

fan::event::task_value_resume_t<bool> process_midi_events(midi_player& player) {
  bool ret = false;

  // First ensure we have data to process
  if (player.parser.size == 0) {
    bool has_data = co_await fill_buffer(player);
    if (!has_data) {
      goto g_return; // No more data to process
    }
  }

  // If header not yet parsed, parse it
  if (!player.header_parsed) {
    int status = midi_parse(&player.parser);
    if (status != MIDI_PARSER_HEADER) {
      throw std::runtime_error("Failed to parse MIDI header");
    }
    player.header = player.parser.header;
    player.header_parsed = true;

    // Update buffer position
    player.buffer_pos = player.buffer.size() - player.parser.size;

    // Need more data?
    if (player.parser.size < 8) {
      bool has_data = co_await fill_buffer(player);
      if (!has_data) {
        goto g_return;
      }
    }
  }

  // Begin track if needed
  if (player.parser.state == MIDI_PARSER_HEADER) {
    int status = midi_parse(&player.parser);
    if (status != MIDI_PARSER_TRACK) {
      goto g_return; // No more tracks
    }
    player.current_track++;

    // Update buffer position
    player.buffer_pos = player.buffer.size() - player.parser.size;
  }

  // Process one event
  if (player.parser.state == MIDI_PARSER_TRACK) {
    // Need more data?
    if (player.parser.size < 3) {
      bool has_data = co_await fill_buffer(player);
      if (!has_data) {
        goto g_return;
      }
    }

    // Process one event
    int status = midi_parse(&player.parser);

    // Update buffer position after parsing
    player.buffer_pos = player.buffer.size() - player.parser.size;

    // Handle event
    if (status == MIDI_PARSER_TRACK_MIDI && player.on_midi_event) {
      player.on_midi_event(player, player.parser.midi);
    }
    else if (status == MIDI_PARSER_TRACK_META && player.on_meta_event) {
      player.on_meta_event(player, player.parser.meta);

      // Check for end of track
      if (player.parser.meta.type == MIDI_META_END_OF_TRACK) {
        // Move to next track
        player.parser.state = MIDI_PARSER_HEADER;
      }
    }
    else if (status == MIDI_PARSER_TRACK_SYSEX && player.on_sysex_event) {
      player.on_sysex_event(player, player.parser.sysex);
    }

    ret = true;
  }

g_return:

  player.time_elapsed = player.current_time_ms - player.last_event_time_ms;
  player.last_event_time_ms = player.current_time_ms;

  co_return ret;
}

std::vector<fan::audio::piece_t> pieces;


struct EventGroup {
  double timestamp_ms;
  std::vector<midi_midi_event> events;
  bool is_chord;
};


std::queue<EventGroup> event_queue;
std::mutex queue_mutex;


void meta_event_cb(midi_player& player, const midi_meta_event& event) {
  if (event.type == MIDI_META_TRACK_NAME && event.length > 0) {
    fan::print("Track name:", std::string((const char*)event.bytes, event.length));
  }
  else if (event.type == MIDI_META_KEY_SIGNATURE && event.length > 0) {
    int8_t key = event.bytes[0];
    uint8_t scale = event.bytes[1]; // 0=major, 1=minor
    int key_index = key + 7;
    fan::print("Key signature", key_names[key_index], scales[scale]);
  } 
  else if (event.type == MIDI_META_SET_TEMPO && event.length >= 3) {
    uint32_t tempo_us_per_quarter = (static_cast<uint32_t>(event.bytes[0]) << 16) |
                                   (static_cast<uint32_t>(event.bytes[1]) << 8) |
                                   static_cast<uint32_t>(event.bytes[2]);
    
    double bpm = 60000000.0 / tempo_us_per_quarter;
    fan::print("Tempo changed to:", bpm, "BPM");
    
    player.tempo_us_per_quarter = tempo_us_per_quarter;
  }
}

void sysex_event_cb(midi_player& player, const midi_sysex_event& event) {
}

void play_event_group(const EventGroup& group);

struct PlaybackState {
  int64_t playback_start_time_us;
  bool is_playing;               
  double current_position_ms;    
  f32_t playback_speed;         
  std::mutex state_mutex;        
};

PlaybackState playback_state = {
  .playback_start_time_us = 0,
  .is_playing = false,
  .current_position_ms = 0,
  .playback_speed = 1.0
};

void midi_timer_callback() {
  std::lock_guard<std::mutex> lock(playback_state.state_mutex);
  
  if (!playback_state.is_playing) {
    return;
  }
  
  int64_t elapsed_us = (fan::time::clock::now() / 1000.0) - playback_state.playback_start_time_us;
  double current_music_time_ms = (elapsed_us / 1000.0) * playback_state.playback_speed;
  playback_state.current_position_ms = current_music_time_ms;
  
  std::vector<EventGroup> events_to_play;
  
  {
    std::lock_guard<std::mutex> queue_lock(queue_mutex);
    while (!event_queue.empty() && event_queue.front().timestamp_ms <= current_music_time_ms) {
      events_to_play.push_back(event_queue.front());
      event_queue.pop();
    }
  }
  
  for (const auto& group : events_to_play) {
    play_event_group(group);
  }
}

void start_playback(double start_position_ms = 0.0) {
  std::lock_guard<std::mutex> lock(playback_state.state_mutex);
  
  playback_state.playback_start_time_us = (fan::time::clock::now() / 1000.0) - 
    static_cast<int64_t>((start_position_ms / playback_state.playback_speed) * 1000);
  playback_state.current_position_ms = start_position_ms;
  playback_state.is_playing = true;
}

void pause_playback() {
  std::lock_guard<std::mutex> lock(playback_state.state_mutex);
  playback_state.is_playing = false;
}

void set_playback_speed(double speed) {
  std::lock_guard<std::mutex> lock(playback_state.state_mutex);
  
  if (playback_state.is_playing) {
    int64_t now = (fan::time::clock::now() / 1000.0);
    int64_t elapsed_us = now - playback_state.playback_start_time_us;
    double old_position_ms = (elapsed_us / 1000.0) * playback_state.playback_speed;
    
    playback_state.playback_start_time_us = now - 
      static_cast<int64_t>((old_position_ms / speed) * 1000);
  }
  
  playback_state.playback_speed = speed;
}

void midi_event_cb(midi_player& player, const midi_midi_event& event) {
  double tick_duration_ms = 0;

  if (player.header.time_division > 0) {
    tick_duration_ms = (double)player.tempo_us_per_quarter / 1000.0 / player.header.time_division;
  }

  player.current_time_ms += player.parser.vtime * tick_duration_ms;
  
  std::lock_guard<std::mutex> lock(queue_mutex);

  // octave
  if ((event_queue.empty() || std::abs(event_queue.back().timestamp_ms - player.current_time_ms) > 1.0)) {
    EventGroup new_group;
    new_group.timestamp_ms = player.current_time_ms;
    new_group.events.push_back(event);
    new_group.is_chord = false;
    event_queue.push(new_group);
  }
  else {
    event_queue.back().events.push_back(event);
    event_queue.back().is_chord = true;
  }
}