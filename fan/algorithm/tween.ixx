module;

export module fan.tween;

import std;
import fan.types;
import fan.math;

export namespace fan::tween {

  namespace easing {
    inline f32_t linear(f32_t t) {
      return t;
    }

    inline f32_t out_elastic(f32_t t) {
      const f32_t c4 = (2.0f * fan::math::pi) / 3.0f;
      return t == 0.0f ? 0.0f : t == 1.0f ? 1.0f :
        std::pow(2.0f, -10.0f * t) * std::sin((t * 10.0f - 0.75f) * c4) + 1.0f;
    }

    inline f32_t out_bounce(f32_t t) {
      const f32_t n1 = 7.5625f;
      const f32_t d1 = 2.75f;
      if (t < 1.0f / d1) {
        return n1 * t * t;
      }
      else if (t < 2.0f / d1) {
        t -= 1.5f / d1;
        return n1 * t * t + 0.75f;
      }
      else if (t < 2.5f / d1) {
        t -= 2.25f / d1;
        return n1 * t * t + 0.9375f;
      }
      else {
        t -= 2.625f / d1;
        return n1 * t * t + 0.984375f;
      }
    }
  }

  template <typename T>
  struct tween_t {
    tween_t(std::function<void(T)> setter, T start, T end, f32_t duration, std::function<f32_t(f32_t)> ease)
        : setter(std::move(setter)), start_val(start), end_val(end), duration(duration), elapsed(0.f), ease_func(std::move(ease)) {}

    bool update(f32_t dt) {
      elapsed += dt;
      if (elapsed >= duration) {
        setter(end_val);
        return true;
      }
      f32_t t = ease_func(elapsed / duration);
      T current = start_val + (end_val - start_val) * t;
      setter(current);
      return false;
    }

    std::function<void(T)> setter;
    T start_val;
    T end_val;
    f32_t duration;
    f32_t elapsed;
    std::function<f32_t(f32_t)> ease_func;
  };

  struct tween_manager_t {
    template <typename T>
    void add(std::function<void(T)> setter, T start_val, T end_val, f32_t duration_sec, std::function<f32_t(f32_t)> ease_func = easing::out_elastic) {
      auto tween = std::make_shared<tween_t<T>>(std::move(setter), start_val, end_val, duration_sec, std::move(ease_func));

      active_tweens.push_back([tween](f32_t dt) {
        return tween->update(dt);
      });
    }

    void update(f32_t dt) {
      for (auto it = active_tweens.begin(); it != active_tweens.end();) {
        if ((*it)(dt)) {
          it = active_tweens.erase(it);
        }
        else {
          ++it;
        }
      }
    }

    std::vector<std::function<bool(f32_t)>> active_tweens;
  };
}