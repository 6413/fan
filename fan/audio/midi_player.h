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

void midi_event_cb(midi_player& player, const midi_midi_event& event) {
  double tick_duration_ms = 0;

  if (player.header.time_division > 0) {
    tick_duration_ms = (double)player.tempo_us_per_quarter / 1000.0 / player.header.time_division;
  }

  player.current_time_ms += player.parser.vtime * tick_duration_ms;
  
  std::lock_guard<std::mutex> lock(queue_mutex);

  // < 15 ms  apart = octave
  if (event_queue.empty() || std::abs(event_queue.back().timestamp_ms - player.current_time_ms) > 2.0) {
    EventGroup new_group;
    new_group.timestamp_ms = player.current_time_ms;
    new_group.events.push_back(event);
    event_queue.push(new_group);
  }
  else {
    event_queue.back().events.push_back(event);
  }

  if (event.status == MIDI_STATUS_NOTE_ON && event.param2 > 0) {
    // Uncomment if needed
    // fan::print("Queued note:", (int)event.param1, "velocity:", (int)event.param2,
    //   "channel:", (int)event.channel, "at time:", player.current_time_ms, "ms", 
    //   "tempo:", current_bpm, "BPM");
  }
}

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

fan::event::task_t process_event_queue() {
  double last_timestamp = 0;
  
  while (true) {
    bool has_events = false;
    EventGroup current_group;
    
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (!event_queue.empty()) {
        has_events = true;
        current_group = event_queue.front();
        event_queue.pop();
      }
    }
    
    if (has_events) {
      // Calculate sleep time based on timestamp difference
      if (last_timestamp > 0) {
        double sleep_time = current_group.timestamp_ms - last_timestamp;
        
        // Use current_bpm for timing, which gets updated whenever tempo changes
        //double time_per_beat_ms = 60000.0 / current_bpm;
        
        // Calculate how long we should actually wait based on current tempo
        double time_adjusted_for_bpm = sleep_time;
        
        if (time_adjusted_for_bpm > 0) {
          co_await fan::co_sleep(time_adjusted_for_bpm);
        }
      }
      
      // Play all events in this group simultaneously
      play_event_group(current_group);
      last_timestamp = current_group.timestamp_ms;
    }
    else {
      // No events to process, yield for a short time
      co_await fan::co_sleep(1);
    }
  }
}