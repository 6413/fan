module;

#include <fan/utility.h>

#ifdef fan_platform_windows
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <Windows.h>
  typedef long(*NtDelayExecution_t)(int Alertable, PLARGE_INTEGER DelayInterval);
  typedef long(* ZwSetTimerResolution_t)(IN ULONG RequestedResolution, IN BOOLEAN Set, OUT PULONG ActualResolution);
  static NtDelayExecution_t NtDelayExecution = (NtDelayExecution_t)(long(__stdcall*)(BOOL, PLARGE_INTEGER)) GetProcAddress(GetModuleHandle("ntdll.dll"), "NtDelayExecution");
  static ZwSetTimerResolution_t ZwSetTimerResolution = (ZwSetTimerResolution_t)(long(__stdcall*)(ULONG, BOOLEAN, PULONG)) GetProcAddress(GetModuleHandle("ntdll.dll"), "ZwSetTimerResolution");
  static void delay_w(float us) {
    static bool once = true;
    if (once) {
      ULONG actualResolution;
      ZwSetTimerResolution(1, true, &actualResolution);
      once = false;
    }
    LARGE_INTEGER interval;
    interval.QuadPart = (LONGLONG)(-10.f * us);
    NtDelayExecution(false, &interval);
  }
#elif defined(fan_platform_unix)
  #include <time.h>
#endif

module fan.time;

import fan.print;

namespace fan {
  namespace time {
    std::uint64_t now() {
    #if defined(fan_platform_windows)
      LARGE_INTEGER freq;
      QueryPerformanceFrequency(&freq);
      double nanoseconds_per_count = 1.0e9 / static_cast<double>(freq.QuadPart);

      LARGE_INTEGER time;
      QueryPerformanceCounter(&time);
      return (std::uint64_t)((double)time.QuadPart * nanoseconds_per_count);
    #elif defined(fan_platform_unix)
      struct timespec t;
      clock_gettime(CLOCK_MONOTONIC, &t);

      return (std::uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
    #endif
    }

    std::uint64_t get_start(){
      static std::uint64_t start = fan::time::now();
      return start;
    }

    f64_t seconds() {
      return fan::time::now() / 1e9;
    }

    void delay(std::uint64_t time) {
    #ifdef fan_platform_windows
      delay_w((float)(time / 1000));
    #elif defined(fan_platform_unix)
      struct timespec t;
      t.tv_sec = time / 1000000000;
      t.tv_nsec = time % 1000000000;
      nanosleep(&t, 0);
    #endif
    }

    scope_timer_print::scope_timer_print() {
      t.start();
    }

    scope_timer_print::~scope_timer_print() {
      std::printf("elapsed: %.6fms\n", t.millis());
    }
  }
  namespace time {
    void timer::start() {
      m_timer = fan::time::now();
      if (m_time == no_interval_v) {
        m_time = infinite_v;
      }
    }
    void timer::start(std::uint64_t time) {
      m_time = time;
      restart();
    }
    void timer::start_seconds(f64_t s) {
      start((std::uint64_t)(s * 1e9));
    }
    void timer::start_millis(f64_t ms) {
      start((std::uint64_t)(ms * 1e6));
    }
    void timer::start_micros(f64_t us) {
      start((std::uint64_t)(us * 1e3));
    }
    void timer::set_time(std::uint64_t time) {
      m_time = time;
    }
    void timer::restart() {
      if (m_time == no_interval_v) {
        return;
      }
      m_timer = fan::time::now();
    }
    timer::timer() {
      start();
    }
    timer::timer(std::uint64_t time, bool start_timer) : m_time(time) {
      if (start_timer) {
        restart();
      }
    }
    timer::timer(bool start_timer) {
      if (start_timer) {
        start();
      }
    }
    bool interval_t::tick(f32_t dt) {
      timer -= dt;
      if (timer <= 0.f) {
        timer += interval;
        return true;
      }
      return false;
    }
    bool& is_measuring() {
      static bool measure = true;
      return measure;
    }
    void set_measuring(bool state) {
      is_measuring() = state;
    }
    void print_measure(const std::string_view msg) {
      if (is_measuring()) {
        fan::print_dbg(msg);
      }
    }
    void print_measure(const std::string_view msg, f64_t time, const std::string_view unit) {
      if (is_measuring()) {
        fan::print_dbg(msg, time, unit);
      }
    }
    void measure(fan::time::timer& timer, const std::string_view msg) {
      if (is_measuring()) {
        fan::print_dbg(msg, "took:", timer.millis(), "ms");
      }
      timer.restart();
    }

    profiler_t global_profiler;

    void profiler_t::begin(std::string_view name) {
      if (!enabled) return;
      entry_t* node;
      if (stack.empty()) {
        node = &roots[name];
      } else {
        node = &stack.back()->children[name];
      }
      node->name = name;
      node->timer.restart();
      stack.push_back(node);

      if (capture_frame_open && capture_warmup_frames == 0) {
        capture_entry_t* capture_node;
        if (capture_stack.empty()) {
          capture_node = &capture_roots[name];
        }
        else {
          capture_node = &capture_stack.back()->children[name];
        }
        capture_node->name = name;
        capture_stack.push_back(capture_node);
      }
    }
    void profiler_t::end(std::string_view name) {
      if (!enabled) return;
      if (stack.empty()) return;
      entry_t* node = stack.back();
      stack.pop_back();
      f64_t elapsed = node->timer.seconds();
      node->accumulated_time += elapsed;
      node->count++;

      if (!capture_stack.empty()) {
        capture_entry_t* capture_node = capture_stack.back();
        capture_stack.pop_back();
        if (capture_record_cpu) {
          capture_node->accumulated_time += elapsed;
          capture_node->count++;
        }
      }
    }
    void profiler_t::add_gpu_time(std::string_view name, f64_t time_ms) {
      if (!enabled) return;
      entry_t* node;
      if (stack.empty()) {
        node = &roots[name];
      } else {
        node = &stack.back()->children[name];
      }
      node->name = name;
      node->accumulated_time += time_ms / 1000.0;
      node->count++;

      if (capture_frame_open && capture_warmup_frames == 0) {
        capture_entry_t* capture_node;
        if (capture_stack.empty()) {
          capture_node = &capture_roots[name];
        }
        else {
          capture_node = &capture_stack.back()->children[name];
        }
        capture_node->name = name;
        capture_node->accumulated_time += time_ms / 1000.0;
        capture_node->count++;
      }
    }
    
    static void update_recursive(profiler_t::entry_t& e) {
      e.last_average = e.count > 0 ? (e.accumulated_time / e.count) * 1000.0 : 0.0;
      e.accumulated_time = 0;
      e.count = 0;
      for (auto& pair : e.children) {
        update_recursive(pair.second);
      }
    }

    void profiler_t::update() {
      if (!enabled) return;
      if (update_timer.seconds() >= 1.0) {
        for (auto& pair : roots) {
          update_recursive(pair.second);
        }
        update_timer.restart();
      }
    }

    bool profiler_t::request_capture(std::uint32_t frame_count, std::function<void()> on_complete) {
      if (frame_count == 0 || capture_active_state) {
        return false;
      }

      capture_roots.clear();
      capture_stack.clear();
      capture_complete_cb = std::move(on_complete);
      capture_target_frames = frame_count;
      capture_frames = 0;
      capture_warmup_frames = 3;
      capture_settle_frames = 0;
      capture_active_state = true;
      capture_frame_open = false;
      capture_record_cpu = false;
      return true;
    }

    void profiler_t::cancel_capture() {
      capture_roots.clear();
      capture_stack.clear();
      capture_complete_cb = {};
      capture_target_frames = 0;
      capture_frames = 0;
      capture_warmup_frames = 0;
      capture_settle_frames = 0;
      capture_active_state = false;
      capture_frame_open = false;
      capture_record_cpu = false;
    }

    bool profiler_t::capture_active() const {
      return capture_active_state;
    }

    void profiler_t::begin_capture_frame() {
      if (!capture_active_state) return;

      capture_frame_open = true;
      capture_stack.clear();
      capture_record_cpu = capture_warmup_frames == 0 && capture_frames < capture_target_frames;
    }

    void profiler_t::end_capture_frame() {
      if (!capture_active_state || !capture_frame_open) return;

      capture_frame_open = false;
      capture_stack.clear();

      if (capture_warmup_frames > 0) {
        --capture_warmup_frames;
        return;
      }

      if (capture_record_cpu) {
        ++capture_frames;
        if (capture_frames == capture_target_frames) {
          capture_record_cpu = false;
          capture_settle_frames = 3;
        }
        return;
      }

      if (capture_settle_frames > 0 && --capture_settle_frames == 0) {
        capture_active_state = false;
        auto on_complete = std::move(capture_complete_cb);
        capture_complete_cb = {};
        if (on_complete) {
          on_complete();
        }
      }
    }

    static void write_capture_entry(std::ostringstream& out, const profiler_t::capture_entry_t& entry, int depth) {
      if (entry.count != 0) {
        out << std::string(depth * 2, ' ') << entry.name << ": "
            << (entry.accumulated_time / entry.count) * 1000.0
            << " ms (" << entry.count << " samples)\n";
      }
      for (const auto& pair : entry.children) {
        write_capture_entry(out, pair.second, depth + (entry.count != 0));
      }
    }

    std::string profiler_t::capture_to_string() const {
      std::ostringstream out;
      out << std::fixed << std::setprecision(6);
      out << "Engine timings\n";
      out << "Frames: " << capture_frames << "\n";
      out << "Average time per profiler sample:\n";
      for (const auto& pair : capture_roots) {
        write_capture_entry(out, pair.second, 0);
      }
      return out.str();
    }

    scope_profiler_t::scope_profiler_t(std::string_view name) : name(name) {
      global_profiler.begin(name);
    }
    scope_profiler_t::~scope_profiler_t() {
      global_profiler.end(name);
    }
  }
  void cooldown_t::tick(f32_t dt) {
    if (current > 0.f) { current -= dt; }
  }
  bool cooldown_t::tick_ready(f32_t dt) {
    tick(dt);
    return is_ready();
  }
}
