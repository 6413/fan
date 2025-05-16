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
    if (key_index >= 0 && key_index < 15) {  // Safety check
      fan::print("Key signature", key_names[key_index], scales[scale]);
    }
  } 
  else if (event.type == MIDI_META_SET_TEMPO && event.length >= 3) {
    uint32_t new_tempo_us_per_quarter = (static_cast<uint32_t>(event.bytes[0]) << 16) |
                                       (static_cast<uint32_t>(event.bytes[1]) << 8) |
                                       static_cast<uint32_t>(event.bytes[2]);
    
    double bpm = 60000000.0 / new_tempo_us_per_quarter;
    fan::print("Tempo changed to:", bpm, "BPM");
    
    // Store the current time before changing tempo
    double old_current_time_ms = player.current_time_ms;
    
    // Update the tempo
    player.tempo_us_per_quarter = new_tempo_us_per_quarter;
    
    // Ensure the tempo change itself is scheduled
    if (player.on_midi_event) {
      // Note: This uses CC 111 which is typically unused, as a marker for tempo change
      midi_midi_event tempo_change_event = {};
      tempo_change_event.status = MIDI_STATUS_CC;
      tempo_change_event.channel = 0;
      tempo_change_event.param1 = 111;  // Special CC number for tempo change
      tempo_change_event.param2 = static_cast<uint8_t>(bpm > 127 ? 127 : bpm);
      
      // Add to event queue with the precise timing
      std::lock_guard<std::mutex> lock(queue_mutex);
      EventGroup tempo_group;
      tempo_group.timestamp_ms = old_current_time_ms;
      tempo_group.events.push_back(tempo_change_event);
      tempo_group.is_chord = false;
      event_queue.push(tempo_group);
    }
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

void set_playback_speed(double speed);

void midi_timer_callback() {
  std::lock_guard<std::mutex> lock(playback_state.state_mutex);
  
  if (!playback_state.is_playing) {
    return;
  }
  
  int64_t now_ns = fan::time::clock::now();
  int64_t start_time_ns = playback_state.playback_start_time_us * 1000;  // Convert us to ns

  int64_t elapsed_ns = now_ns - start_time_ns;
  double elapsed_ms = elapsed_ns / 1'000'000.0;
  
  double current_music_time_ms = elapsed_ms * playback_state.playback_speed;
  playback_state.current_position_ms = current_music_time_ms;
  
  std::vector<EventGroup> events_to_play;
  
  {
    std::lock_guard<std::mutex> queue_lock(queue_mutex);
    
    // Process all events that should have played by now
    while (!event_queue.empty() && event_queue.front().timestamp_ms <= current_music_time_ms) {
      events_to_play.push_back(event_queue.front());
      event_queue.pop();
    }
    
    if (event_queue.size() > 1000) {
    //  fan::print("Warning: Large event queue size:", event_queue.size());
    }
  }
  
  for (const auto& group : events_to_play) {
    bool has_tempo_change = false;
    for (const auto& event : group.events) {
      if (event.status == MIDI_STATUS_CC && event.param1 == 111) {
        has_tempo_change = true;
        
        double original_speed = playback_state.playback_speed;
        
        // This playback speed adjustment is optional - uncomment if you
        // want tempo changes in the MIDI to affect the playback speed
        
         double scale_factor = event.param2 / 120.0;  // Normalize around 120 BPM
         set_playback_speed(1.0 * scale_factor);
        
        break;
      }
    }
    
    if (!has_tempo_change) {
      play_event_group(group);
    }
  }
}

void start_playback(double start_position_ms = 0.0) {
  std::lock_guard<std::mutex> lock(playback_state.state_mutex);
  
  int64_t now_ns = fan::time::clock::now();
  int64_t offset_ns = static_cast<int64_t>((start_position_ms / playback_state.playback_speed) * 1'000'000);

  playback_state.playback_start_time_us = (now_ns - offset_ns) / 1000;
  playback_state.current_position_ms = start_position_ms;

  playback_state.is_playing = true;
}

void pause_playback() {
  std::lock_guard<std::mutex> lock(playback_state.state_mutex);
  playback_state.is_playing = false;
}

void set_playback_speed(double speed) {
  
  if (playback_state.is_playing) {
    int64_t now_ns = fan::time::clock::now();
    int64_t elapsed_ns = now_ns - playback_state.playback_start_time_us * 1000; // Convert stored us to ns
    
    double old_position_ms = (elapsed_ns / 1'000'000.0) * playback_state.playback_speed;

    int64_t new_start_time_ns = now_ns - static_cast<int64_t>((old_position_ms / speed) * 1'000'000);

    playback_state.playback_start_time_us = new_start_time_ns / 1000;  // Store in microseconds
  }

  playback_state.playback_speed = speed;
}


void midi_event_cb(midi_player& player, const midi_midi_event& event) {
  double tick_duration_ms = 0;

  if (player.header.time_division > 0) {
    tick_duration_ms = (double)player.tempo_us_per_quarter / 1000.0 / player.header.time_division;
  }

  player.current_time_ms += player.parser.vtime * tick_duration_ms;
  
  if (event.status == MIDI_STATUS_CC && event.param1 == 64) {
    EventGroup pedal_group;
    pedal_group.timestamp_ms = player.current_time_ms;
    pedal_group.events.push_back(event);
    pedal_group.is_chord = false;
    
    std::lock_guard<std::mutex> lock(queue_mutex);
    event_queue.push(pedal_group);
    return;
  }
  
  std::lock_guard<std::mutex> lock(queue_mutex);

  if ((event_queue.empty() || std::abs(event_queue.back().timestamp_ms - player.current_time_ms) > 15.0)) {
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