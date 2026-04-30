module;

#include <fan/utility.h>

export module fan.time;

import std;

import fan.types;

export namespace fan {
  namespace time {
    std::uint64_t now();
    f64_t seconds();
    void delay(std::uint64_t time);

    struct timer {
      timer() = default;
      explicit timer(std::uint64_t time, bool start_timer) {
        if (start_timer) {
          start(time);
        }
      }
      explicit timer(f64_t time, bool start_timer) {
        if (start_timer) {
          start(time);
        }
      }
      explicit timer(bool start_timer) {
        if (start_timer) {
          start();
        }
      }
      constexpr std::uint64_t count() const {
        return m_time;
      }
      constexpr std::uint64_t duration() const {
        return count();
      }
      constexpr f64_t duration_seconds() const {
        return duration() / 1e9;
      }
      void start() {
        m_timer = fan::time::now();
        m_time = -2;
      }
      void start(std::uint64_t time) {
        this->start();
        m_time = time;
      }
      void start_seconds(f64_t s) {
        start();
        m_time = (std::uint64_t)(s * 1e9);
      }
      void start_millis(f64_t ms) {
        start();
        m_time = (std::uint64_t)(ms * 1e6);
      }
      void start_micros(f64_t us) {
        start();
        m_time = (std::uint64_t)(us * 1e3);
      }

      void set_time(std::uint64_t time) {
        m_time = time;
      }
      void restart() {
        auto prev = m_time;
        start();
        m_time = prev;
      }
      bool finished() const {
        auto elapsed = this->elapsed();
        return elapsed >= m_time;
      }
      explicit operator bool() const {
        return finished();
      }
      bool started() const {
        return m_time != (std::uint64_t)-1;
      }
      std::uint64_t elapsed() const {
        return m_timer == 0 ? 0 : fan::time::now() - m_timer;
      }
      double seconds() const {
        return elapsed() / 1e9;
      }
      double millis() const {
        return elapsed() / 1e6;
      }

      std::uint64_t start_time() const {
        return m_timer;
      }
      std::uint64_t start_time_seconds() const {
        return m_timer / 1e9;
      }

      std::uint64_t m_timer = 0;
      std::uint64_t m_time = (std::uint64_t)-1;
    };

    timer seconds_timer(f64_t s) {
      timer t;
      t.start_seconds(s);
      return t;
    }

    timer millis_timer(f64_t ms) {
      timer t;
      t.start_millis(ms);
      return t;
    }

    timer micros_timer(f64_t us) {
      timer t;
      t.start_micros(us);
      return t;
    }

    template <typename F>
    struct scope_timer {
      fan::time::timer t;
      F cb;
      scope_timer(F cb_) : cb(static_cast<F&&>(cb_)) {
        t.start();
      }
      ~scope_timer() {
        cb(t);
      }
    };

    struct scope_timer_print {
      scope_timer_print();
      scope_timer_print(const scope_timer_print&) = delete;
      scope_timer_print(scope_timer_print&&) = delete;
      ~scope_timer_print();
      fan::time::timer t;
    };

    template<FAN_UNIQUE_CALL>
    bool every(f64_t interval_ms) {
      static std::uint64_t last_time = 0;  
      std::uint64_t interval_ns = (std::uint64_t)(interval_ms * 1e6);
      std::uint64_t current = fan::time::now();
      std::uint64_t elapsed = current - last_time;
      std::uint64_t finished = (last_time == 0) | (elapsed >= interval_ns);
      last_time = last_time * (finished ^ 1) + current * finished;
      return finished;
    }

    struct interval_t {
      interval_t() = default;
      interval_t(f32_t interval) : interval(interval) {}
      bool tick(f32_t dt) {
        timer -= dt;
        if (timer <= 0.f) {
          timer += interval;
          return true;
        }
        return false;
      }
      void reset() { timer = 0.f; }
      f32_t interval = 0.f;
      f32_t timer = 0.f;
    };
  }

  struct cooldown_t {
    f32_t max = 0.f;
    f32_t current = 0.f;

    void tick(f32_t dt) {
      if (current > 0.f) { current -= dt; }
    }

    template <typename F>
    bool tick_and_fire(f32_t dt, F&& cb) {
      tick(dt);
      if (!is_ready()) return false;
      reset();
      cb();
      return true;
    }

    bool is_ready() const { return current <= 0.f; }
    void reset() { current = max; }
    void expire() { current = 0.f; }
    bool tick_ready(f32_t dt) {
      tick(dt);
      return is_ready();
    }
    static cooldown_t full(f32_t max) { cooldown_t c{max}; c.current = max; return c; }
  };
}