#define _INCLUDE_TOKEN(p0, p1) <p0/p1>

#ifndef WITCH_INCLUDE_PATH
  #define WITCH_INCLUDE_PATH WITCH
#endif
#ifndef FAN_INCLUDE_PATH
  #define FAN_INCLUDE_PATH /usr/local/include
#endif

#include _INCLUDE_TOKEN(WITCH_INCLUDE_PATH,WITCH.h)
#include _INCLUDE_TOKEN(FAN_INCLUDE_PATH,fan/types/types.h)

#include _WITCH_PATH(IO/IO.h)
#include _WITCH_PATH(FS/FS.h)
#include _WITCH_PATH(A/A.h)
#include _WITCH_PATH(IO/print.h)
#include _WITCH_PATH(TH/TH.h)

#include _FAN_PATH(audio/audio.h)

#include <fftw/fftw3.h>

#include <fan/pch.h>

void WriteOut(const char *format, ...){
	IO_fd_t fd_stdout;
	IO_fd_set(&fd_stdout, FD_OUT);
	va_list argv;
	va_start(argv, format);
	IO_vprint(&fd_stdout, format, argv);
	va_end(argv);
}

int main(int argc, char **argv){
  if(argc != 2){
    return 0;
  }

  loco_t loco;

  struct data_t {
    fan::system_audio_t system_audio;
    std::vector<f32_t> audio_data;
    std::mutex mut;
  }d;


  if(d.system_audio.Open() != 0){
    __abort();
  }

  fan::audio_t audio;
  audio.bind(&d.system_audio);

  fan::audio_t::piece_t piece;
  sint32_t err = audio.Open(&piece, "audio/w_voice.sac", 0);
  if(err != 0){
    WriteOut("piece open failed %lx\n", err);
    return 0;
  }

  {
    fan::audio_t::PropertiesSoundPlay_t p;
    p.Flags.Loop = true;
    p.GroupID = 0;
    audio.SoundPlay(&piece, &p);
  }

  auto smoke_texture = loco.image_load("images/smoke.webp");

  fan::vec2 window_size = loco.window.get_size();
  loco_t::particles_t::properties_t p;
  p.position = fan::vec3(window_size.x / 2, window_size.y / 2, 10);
  p.count = 8;
  p.size = 30;
  p.begin_angle = 0;
  p.end_angle = 1.0;
  p.alive_time = 3e+9;
  p.position_velocity = fan::vec2(0, 100);
  p.image = smoke_texture;
  p.color = fan::color(0.4, 0.4, 1.4);


  std::vector<f32_t> currentHeights;

  IO_fd_t fd_in;
  IO_fd_set(&fd_in, STDIN_FILENO);



  d.system_audio.Process.ResultFramesCB = [](fan::system_audio_t::Process_t* Process, float* samples, uint32_t samplesi) {
    fan::system_audio_t* system_audio = OFFSETLESS(Process, fan::system_audio_t, Process);
    data_t* data = OFFSETLESS(system_audio, data_t, system_audio);
    std::lock_guard<std::mutex> lock(data->mut);
    data->audio_data.clear();
    data->audio_data.insert(data->audio_data.end(), samples, samples + samplesi / 2);
  };

  std::vector<loco_t::shape_t> shapes;
  loco.loop([&] {
    uint8_t buffer[0x200];
    IO_ssize_t r = IO_read(&fd_in, buffer, sizeof(buffer));
    if (r < 0) {
      //break;
      return;
    }

    std::lock_guard<std::mutex> lock(d.mut);
    int windowSize = d.audio_data.size();


    //shapes.resize(windowSize, p);
    shapes.clear();
    for (int i = 0; i < windowSize; ++i) {
      p.position.z = windowSize -i;
      shapes.push_back(p);
    }


    // Create FFTW plan
    fftw_complex* fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * windowSize);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * windowSize);
    fftw_plan fft_plan = fftw_plan_dft_1d(windowSize, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);



    std::vector<double> window(windowSize, 1.0);


    // Copy window of audio data to FFTW input
    for (int i = 0; i < windowSize; i++) {
      fft_in[i][0] = d.audio_data[i] * window[i];  // Apply window
      fft_in[i][1] = 0.0;
    }


    if (fft_plan != nullptr) {
      // Execute FFT
      fftw_execute(fft_plan);

    }


    // Calculate magnitudes and convert to decibels
    std::vector<float> magnitudes;
    for (int i = 0; i < windowSize; i++) {  // Use first half of FFT output
      float magnitude = sqrt(fft_out[i][0] * fft_out[i][0] + fft_out[i][1] * fft_out[i][1]);
      float db = 20 * log10(magnitude);
      magnitudes.push_back(abs(db));  // Take absolute value
    }

    float itemSpacing = 5;  // Set the spacing between items

    ImVec2 startPos = ImGui::GetCursorPos();  // Get the starting position
    //shapes.clear();

    // Initialize current heights if not done already
    if (currentHeights.size() != magnitudes.size()) {
      currentHeights = std::vector<float>(magnitudes.size(), 0);
    }

    for (int i = 0; i < magnitudes.size(); i += 2) {
      if (std::isinf(magnitudes[i])) {
        continue;
      }

      fan::vec2 window_size = loco.window.get_size();



      // Calculate target height and interpolate
      float targetHeight = 100.0 - magnitudes[i] * 4;
      float interpolationSpeed = 0.5;  // Adjust this value to change the speed of interpolation
      currentHeights[i] = currentHeights[i] + (targetHeight - currentHeights[i]) * interpolationSpeed;

      fan::vec2 box_size(window_size.x  / magnitudes.size(), currentHeights[i]);
      f32_t padding = 13 + box_size.x * 1.5;
      //    padding += box_size.x;
      fan::vec2 rpos;
      rpos.x = window_size.x / magnitudes.size() / box_size.x + i * box_size.x + padding * i;
      rpos.x /= 2;
      rpos.y = window_size.y - box_size.x;

      auto& ri = *(loco_t::particles_t::ri_t*)loco.shaper.GetData(shapes[i]);

      ri.position = rpos;
      ri.size = box_size.x / 2;
      ri.color = fan::color::hsv((1.0 - i / (f32_t)magnitudes.size()) * 360.f, 100, 100);
      ri.count = box_size.x * 5000;
      ri.position_velocity.y = box_size.y;
    }

    fftw_destroy_plan(fft_plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
  });

  audio.unbind();

  d.system_audio.Close();

  return 0;
}
