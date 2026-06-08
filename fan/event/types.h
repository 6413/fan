#pragma once

// this file is intended for macros

// executes given code every 'time_ms'
// destroys when out of scope
#define fan_ev_timer(time_ms, code) \
  auto CONCAT(timer__var__, __COUNTER__) = fan::event::task_timer(time_ms, [&]() -> bool {code return false; })
#define fan_ev_timer_loop(time_ms, code) \
  [&]{ \
    static fan::time::timer c{(uint64_t)time_ms * (uint64_t)1e+6, true}; \
    if (c.finished()) { \
      code \
      c.restart(); \
    } \
  }()
#define fan_ev_timer_loop_init(time_ms, condition, code) \
  [&]{ \
    static fan::time::timer c{(uint64_t)time_ms * (uint64_t)1e+6, true}; \
    if (c.finished() || condition) { \
      code \
      c.restart(); \
    } \
  }()