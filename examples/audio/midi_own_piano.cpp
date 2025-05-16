
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


#include <iostream>
#include <windows.h>
#include <mmsystem.h>
#include <iomanip>
#include <dbt.h>
#include <thread>

#include <fan/types/types.h>
#include <fan/math/math.h>
#include <fan/time/timer.h>
#include <fan/imgui/implot.h>

import fan;

#include <fan/graphics/types.h>

#include <fan/audio/midi_parser.h>
#include <fan/audio/midi_player.h>


#pragma comment(lib, "winmm.lib")

// Piano key information structure
struct PianoKey {
  int midiNote;
  std::string noteName;
  bool isPressed;
};

// Global variables
std::vector<PianoKey> piano(88);
std::map<UINT, HMIDIIN> openMidiDevices;
std::mutex deviceMutex;
bool isRunning = true;
HWND hWnd = NULL;

#define WM_MIDIDEVICE_CHANGE (WM_USER + 1)

std::string getNoteNameFromNumber(int midiNote) {
  const char* noteNames[] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
  int octave = (midiNote / 12) - 1;
  int noteInOctave = midiNote % 12;
  return noteNames[noteInOctave] + std::to_string(octave);
}

void initializePiano() {
  for (int i = 0; i < 88; i++) {
    int midiNote = i + 21;
    piano[i].midiNote = midiNote;
    piano[i].noteName = getNoteNameFromNumber(midiNote);
    piano[i].isPressed = false;
  }
}

void displayPianoState() {

  std::lock_guard<std::mutex> lock(deviceMutex);
  std::cout << "Connected MIDI devices: " << openMidiDevices.size() << std::endl;
  for (const auto& device : openMidiDevices) {
    MIDIINCAPS midiInCaps;
    midiInGetDevCaps(device.first, &midiInCaps, sizeof(MIDIINCAPS));
    std::cout << "- [" << device.first << "] " << midiInCaps.szPname << std::endl;
  }
  std::cout << std::endl;

  std::cout << "Currently pressed keys:\n";
  bool anyKeyPressed = false;

  for (const auto& key : piano) {
    if (key.isPressed) {
      anyKeyPressed = true;
      std::cout << std::setw(5) << key.noteName;
    }
  }

  if (!anyKeyPressed) {
    std::cout << "(none)";
  }

  std::cout << "\n\nRecent MIDI activity:\n";
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
}visual_piano;


void play_event_group(const EventGroup& group) {

}

void CALLBACK MidiInProc(HMIDIIN hMidiIn, UINT wMsg, DWORD_PTR dwInstance, DWORD_PTR dwParam1, DWORD_PTR dwParam2) {
  if (wMsg == MIM_DATA) {
    BYTE status = (BYTE)(dwParam1 & 0xFF);
    BYTE data1 = (BYTE)((dwParam1 >> 8) & 0xFF);  // Note number
    BYTE data2 = (BYTE)((dwParam1 >> 16) & 0xFF); // Velocity

    BYTE msgType = status & 0xF0;

    if (data1 >= 21 && data1 <= 108) {
      int pianoIndex = data1 - 21;

      if (msgType == 0x90 && data2 > 0) {  // Note On
        piano[pianoIndex].isPressed = true;
        std::cout << "Note On: " << piano[pianoIndex].noteName
          << " (MIDI: " << (int)data1 << ") Velocity: " << (int)data2 << std::endl;
        fan::audio::play(pieces[pianoIndex]);
        visual_piano.keys[pianoIndex].visual.set_color(fan::color::rgb(168, 59, 59));
      }
      else if (msgType == 0x80 || (msgType == 0x90 && data2 == 0)) {  // Note Off
        piano[pianoIndex].isPressed = false;
        std::cout << "Note Off: " << piano[pianoIndex].noteName
          << " (MIDI: " << (int)data1 << ")" << std::endl;
        int note_in_octave = pianoIndex % 12;

        bool is_black = (
          note_in_octave == 1 ||  // A#
          note_in_octave == 4 ||  // C#
          note_in_octave == 6 ||  // D#
          note_in_octave == 9 ||  // F#
          note_in_octave == 11     // G#
          );
        if (is_black) {
          visual_piano.keys[pianoIndex].visual.set_color(0);
        }
        else {
          visual_piano.keys[pianoIndex].visual.set_color(1);
        }
      }

      displayPianoState();
    }
  }
  else if (wMsg == MIM_OPEN) {
    std::cout << "MIDI device opened successfully" << std::endl;
  }
  else if (wMsg == MIM_CLOSE) {
    std::cout << "MIDI device closed" << std::endl;
  }
  else if (wMsg == MIM_ERROR) {
    std::cout << "MIDI device error" << std::endl;
  }
}

bool openMidiDevice(UINT deviceID) {
  std::lock_guard<std::mutex> lock(deviceMutex);

  if (openMidiDevices.find(deviceID) != openMidiDevices.end()) {
    return true;
  }

  HMIDIIN hMidiIn = NULL;
  MMRESULT result = midiInOpen(&hMidiIn, deviceID, (DWORD_PTR)(void*)MidiInProc, 0, CALLBACK_FUNCTION);

  if (result != MMSYSERR_NOERROR) {
    std::cout << "Failed to open MIDI device " << deviceID << ". Error code: " << result << std::endl;
    return false;
  }

  result = midiInStart(hMidiIn);
  if (result != MMSYSERR_NOERROR) {
    std::cout << "Failed to start MIDI input for device " << deviceID << ". Error code: " << result << std::endl;
    midiInClose(hMidiIn);
    return false;
  }

  openMidiDevices[deviceID] = hMidiIn;

  MIDIINCAPS midiInCaps;
  midiInGetDevCaps(deviceID, &midiInCaps, sizeof(MIDIINCAPS));
  std::cout << "Successfully connected to MIDI device: " << midiInCaps.szPname << std::endl;

  return true;
}

void closeMidiDevice(UINT deviceID) {
  std::lock_guard<std::mutex> lock(deviceMutex);

  auto it = openMidiDevices.find(deviceID);
  if (it != openMidiDevices.end()) {
    HMIDIIN hMidiIn = it->second;
    midiInStop(hMidiIn);
    midiInClose(hMidiIn);
    openMidiDevices.erase(it);

    std::cout << "Disconnected MIDI device ID: " << deviceID << std::endl;
  }
}

void closeAllMidiDevices() {
  std::lock_guard<std::mutex> lock(deviceMutex);

  for (auto& device : openMidiDevices) {
    HMIDIIN hMidiIn = device.second;
    midiInStop(hMidiIn);
    midiInClose(hMidiIn);
  }

  openMidiDevices.clear();
}

void scanAndOpenMidiDevices() {
  UINT numDevs = midiInGetNumDevs();
  std::cout << "Scanning for MIDI Input Devices: Found " << numDevs << std::endl;

  for (UINT i = 0; i < numDevs; i++) {
    MIDIINCAPS midiInCaps;
    midiInGetDevCaps(i, &midiInCaps, sizeof(MIDIINCAPS));
    std::cout << "[" << i << "] " << midiInCaps.szPname << std::endl;

    openMidiDevice(i);
  }

  if (numDevs == 0) {
    std::cout << "No MIDI devices found. Connect a device and it will be detected automatically." << std::endl;
  }

  displayPianoState();
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  switch (uMsg) {
  case WM_DEVICECHANGE: {
    if (wParam == DBT_DEVICEARRIVAL || wParam == DBT_DEVICEREMOVECOMPLETE) {
      PostMessage(hwnd, WM_MIDIDEVICE_CHANGE, 0, 0);
    }
    break;
  }
  case WM_DESTROY:
    PostQuitMessage(0);
    break;
  default:
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
  }
  return 0;
}

HWND createHiddenWindow() {
  WNDCLASSEX wc = { 0 };
  wc.cbSize = sizeof(WNDCLASSEX);
  wc.lpfnWndProc = WindowProc;
  wc.hInstance = GetModuleHandle(NULL);
  wc.lpszClassName = "MidiDeviceNotification";

  RegisterClassEx(&wc);

  // Create the window
  HWND hwnd = CreateWindowEx(
    0,
    "MidiDeviceNotification",
    "Midi Device Notification Window",
    0,
    0, 0, 0, 0,
    HWND_MESSAGE,
    NULL,
    GetModuleHandle(NULL),
    NULL
  );

  return hwnd;
}

void registerForDeviceNotifications(HWND hwnd) {
  DEV_BROADCAST_DEVICEINTERFACE notificationFilter = { 0 };
  notificationFilter.dbcc_size = sizeof(DEV_BROADCAST_DEVICEINTERFACE);
  notificationFilter.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE;

  HDEVNOTIFY hDevNotify = RegisterDeviceNotification(
    hwnd,
    &notificationFilter,
    DEVICE_NOTIFY_WINDOW_HANDLE | DEVICE_NOTIFY_ALL_INTERFACE_CLASSES
  );

  if (hDevNotify == NULL) {
    std::cout << "Failed to register for device notifications. Error: " << GetLastError() << std::endl;
  }
}

void messageLoop() {
  MSG msg;
  while (GetMessage(&msg, NULL, 0, 0) > 0) {
    if (msg.message == WM_MIDIDEVICE_CHANGE) {
      std::cout << "\nMIDI device change detected - rescanning devices...\n" << std::endl;
      scanAndOpenMidiDevices();
    }
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}

// SIGINT handler
BOOL WINAPI ConsoleHandler(DWORD signal) {
  if (signal == CTRL_C_EVENT) {
    isRunning = false;
    PostQuitMessage(0);
    return TRUE;
  }
  return FALSE;
}

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

std::mutex mut;

struct key_info_t {
  int position = 0;
  f32_t velocity = 0;
  fan::audio_t::SoundPlayID_t play_id;
};

std::unordered_map<fan::audio_t::_piece_t*, std::vector<key_info_t>> key_info;

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

  f32_t volume = fan::audio::get_volume();

  visual_piano.init(engine);

  SetConsoleCtrlHandler(ConsoleHandler, TRUE);

  initializePiano();

  hWnd = createHiddenWindow();
  if (hWnd == NULL) {
    std::cout << "Failed to create notification window. Error: " << GetLastError() << std::endl;
    return 1;
  }

  registerForDeviceNotifications(hWnd);

  scanAndOpenMidiDevices();

  std::cout << "\nWaiting for MIDI input\n";

  std::thread msg_thread(messageLoop);

  apply_envelope(engine);


  engine.loop([&] {

    fan_graphics_gui_window("audio controls") {
      
      if (fan::graphics::gui::drag_float("bpm", &playback_state.playback_speed, 0.01, 0.01)) {

      }

      if (fan::graphics::gui::drag_float("volume", &volume, 0.01f, 0.0f, 1.0f)) {
        fan::audio::set_volume(volume);
      }
    }
    
    // iterate by depth
    for (auto [i, key] : fan::enumerate(visual_piano.keys)) {
      if (fan::window::is_mouse_clicked() && key.visual.is_mouse_inside()) {
        auto found = key_info.find(pieces[i]._piece);
        if (found == key_info.end()) {
          fan::throw_error("AA");
        }

        found->second.push_back({});
        auto& sound = found->second.back();
        sound.velocity = 0.7;
        sound.play_id = fan::audio::play(pieces[i]);
        sound.position = 0;
        break;
        //visual_piano.keys[pianoIndex].visual.set_color(fan::color::rgb(168, 59, 59));
      }
      else if (fan::window::is_mouse_released() && key.visual.is_mouse_inside()) {
        break;
      }
    }

    if (ImPlot::BeginPlot("pcm")) {
      ImPlot::PlotLine("pcm0", engine.system_audio.Out.frames[0], fan::system_audio_t::_constants::CallFrameCount * fan::system_audio_t::_constants::ChannelAmount);
      ImPlot::PlotLine("pcm1", engine.system_audio.Out.frames[1], fan::system_audio_t::_constants::CallFrameCount * fan::system_audio_t::_constants::ChannelAmount);
      ImPlot::EndPlot();
    }
  });
  

  
  closeAllMidiDevices();

  if (msg_thread.joinable()) {
    msg_thread.join();
  }

  return 0;
}