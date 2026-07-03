module;

export module fan.tween;

import std;
import fan.types;
import fan.math;

export namespace fan::tween {

  namespace easing {
    inline float linear(float t) {
      return t;
    }

    inline float out_elastic(float t) {
      const float c4 = (2.0f * fan::math::pi) / 3.0f;
      return t == 0.0f ? 0.0f : t == 1.0f ? 1.0f :
        std::pow(2.0f, -10.0f * t) * std::sin((t * 10.0f - 0.75f) * c4) + 1.0f;
    }

    inline float out_bounce(float t) {
      const float n1 = 7.5625f;
      const float d1 = 2.75f;
      if (t < 1.0f / d1) {
        return n1 * t * t;
      } else if (t < 2.0f / d1) {
        t -= 1.5f / d1;
        return n1 * t * t + 0.75f;
      } else if (t < 2.5f / d1) {
        t -= 2.25f / d1;
        return n1 * t * t + 0.9375f;
      } else {
        t -= 2.625f / d1;
        return n1 * t * t + 0.984375f;
      }
    }
  }

  template <typename T>
  struct tween_t {
    std::function<void(T)> setter;
    T start_val;
    T end_val;
    float duration;
    float elapsed;
    std::function<float(float)> ease_func;

    tween_t(std::function<void(T)> setter, T start, T end, float duration, std::function<float(float)> ease)
        : setter(std::move(setter)), start_val(start), end_val(end), duration(duration), elapsed(0.f), ease_func(std::move(ease)) {}

    bool update(float dt) {
      elapsed += dt;
      if (elapsed >= duration) {
        setter(end_val);
        return true;
      }
      float t = ease_func(elapsed / duration);
      T current = start_val + (end_val - start_val) * t;
      setter(current);
      return false;
    }
  };

  struct tween_manager_t {
    std::vector<std::function<bool(float)>> active_tweens;

    template <typename T>
    void add(std::function<void(T)> setter, T start_val, T end_val, float duration_sec, std::function<float(float)> ease_func = easing::out_elastic) {
      auto tween = std::make_shared<tween_t<T>>(std::move(setter), start_val, end_val, duration_sec, std::move(ease_func));
      
      active_tweens.push_back([tween](float dt) {
        return tween->update(dt);
      });
    }

    void update(float dt) {
      for (auto it = active_tweens.begin(); it != active_tweens.end();) {
        if ((*it)(dt)) {
          it = active_tweens.erase(it);
        } else {
          ++it;
        }
      }
    }
  };
}
