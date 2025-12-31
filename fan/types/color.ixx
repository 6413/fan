module;

//#undef FAN_GUI

#include <string>
#include <cmath>
#include <algorithm>

export module fan.types.color;

import fan.types;
import fan.types.vector;

#if defined(FAN_GUI)
import fan.graphics.gui.types;
#endif

#pragma pack(push, 1)

export namespace fan {
  // internal format 0-1
  struct color {
    constexpr color() = default;
    template <typename T>
    constexpr color(const fan::vec3_wrap_t<T>& v) {
      r = v.x;
      g = v.y;
      b = v.z;
    }
    template <typename T>
    constexpr color(const fan::vec4_wrap_t<T>& v) {
      *(fan::vec4*)this = v;
    }
    constexpr color(f32_t r, f32_t g, f32_t b, f32_t a = 1) : r(r), g(g), b(b), a(a) {
      this->r = r;
      this->g = g;
      this->b = b;
      this->a = a;
    }
    constexpr color(uint8_t* begin, uint8_t* end) {
      uint8_t* ptr = begin;
      int i = 0;
      while (ptr != end) {
        (*this)[i++] = *ptr / f32_t(255);
        ++ptr;
      }
    }
    constexpr color(f32_t value) : r(0), g(0), b(0), a(0) {
      this->r = value;
      this->g = value;
      this->b = value;
      this->a = value;
    }
    constexpr color& operator&=(const color& c) {
      r = (unsigned int)r & (unsigned int)c.r;
      g = (unsigned int)g & (unsigned int)c.g;
      b = (unsigned int)b & (unsigned int)c.b;
      a = (unsigned int)a & (unsigned int)c.a;
      return *this;
    }
    constexpr color& operator^=(const color& c) {
      r = (int)r ^ (int)c.r;
      g = (int)g ^ (int)c.g;
      b = (int)b ^ (int)c.b;
      return *this;
    }
    constexpr bool operator==(const color& c) const {
      return r == c.r && g == c.g && b == c.b && a == c.a;
    }
    constexpr bool operator!=(const color& c) const {
      return !(*this == c);
    }
    constexpr f32_t& operator[](size_t x) {
      return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
    }
    constexpr f32_t operator[](size_t x) const {
      return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
    }
    constexpr color operator-=(const color& c) {
      return color(r -= c.r, g -= c.g, b -= c.b, a -= c.a);
    }
    constexpr color operator-() const {
      return color(-r, -g, -b, a);
    }
    constexpr color operator-(const color& c) const {
      return color(r - c.r, g - c.g, b - c.b, c.a != 1 ? a - c.a : a);
    }
    constexpr color operator+(const color& c) const {
      return color(r + c.r, g + c.g, b + c.b, a + c.a);
    }
    template <typename T>
    constexpr color operator*(T value) const {
      return color(r * value, g * value, b * value, a * value);
    }
    constexpr color operator*(const color& value) const {
      return color(r * value.r, g * value.g, b * value.b, a);
    }
    template <typename T>
    constexpr color operator/(T value) const {
      return color(r / value, g / value, b / value);
    }

    template <typename T>
    constexpr color mult_no_alpha(T value) const {
      return color(r * value, g * value, b * value);
    }
    f32_t* data() {
      return &r;
    }

    static constexpr f32_t clamp(f32_t value, f32_t min, f32_t max) {
      return value < min ? min : (value > max ? max : value);
    }

    static constexpr uint32_t to_byte(f32_t value) {
      return static_cast<uint32_t>(color::clamp(value, 0.0f, 1.0f) * 255);
    }

    static constexpr uint32_t pack_color(f32_t c1, f32_t c2, f32_t c3, f32_t c4) {
      return (to_byte(c1) << 24) | (to_byte(c2) << 16) | (to_byte(c3) << 8) | to_byte(c4);
    }
    static constexpr void unpack_color(uint32_t color_, f32_t& c1, f32_t& c2, f32_t& c3, f32_t& c4) {
      c1 = ((color_ >> 24) & 0xFF) / 255.0f;
      c2 = ((color_ >> 16) & 0xFF) / 255.0f;
      c3 = ((color_ >> 8) & 0xFF) / 255.0f;
      c4 = (color_ & 0xFF) / 255.0f;
    }

    constexpr uint32_t get_rgba() const { return pack_color(r, g, b, a); }
    constexpr uint32_t get_abgr() const { return pack_color(a, b, g, r); }
    constexpr uint32_t get_argb() const { return pack_color(a, r, g, b); }
    constexpr uint32_t get_bgra() const { return pack_color(b, g, r, a); }

    constexpr void set_rgba(uint32_t color_) { unpack_color(color_, r, g, b, a); }
    constexpr void set_abgr(uint32_t color_) { unpack_color(color_, a, b, g, r); }
    constexpr void set_argb(uint32_t color_) { unpack_color(color_, a, r, g, b); }
    constexpr void set_bgra(uint32_t color_) { unpack_color(color_, b, g, r, a); }

    static constexpr fan::color from_rgba(uint32_t color_) {
      fan::color c;
      c.set_rgba(color_);
      return c;
    }
    static constexpr fan::color from_abgr(uint32_t color_) {
      fan::color c;
      c.set_abgr(color_);
      return c;
    }
    static constexpr fan::color from_argb(uint32_t color_) {
      fan::color c;
      c.set_argb(color_);
      return c;
    }
    static constexpr fan::color from_bgra(uint32_t color_) {
      fan::color c;
      c.set_bgra(color_);
      return c;
    }

    static constexpr fan::color from_rgb(uint32_t color_) {
      fan::color c;
      uint32_t rgba_color = (color_ << 8) | 0xFF;
      c.set_rgba(rgba_color);
      return c;
    }

    static inline f32_t srgb_to_linear_channel(f32_t c) {
      if (c <= 0.04045f) {
        return c / 12.92f;
      }
      return std::pow((c + 0.055f) / 1.055f, 2.4f);
    }
    static inline f32_t linear_to_srgb_channel(f32_t c) {
      if (c <= 0.0031308f) {
        return c * 12.92f;
      }
      return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    }
    static color to_srgb(const color& c);
    static color to_linear(const color& c);
    static void rgb_to_hsl(f32_t r, f32_t g, f32_t b, f32_t& h, f32_t& s, f32_t& l);
    static constexpr f32_t hue_to_rgb(f32_t p, f32_t q, f32_t t) {
      if (t < 0.0f) {
        t += 1.0f;
      }
      if (t > 1.0f) {
        t -= 1.0f;
      }
      if (t < 1.0f/6.0f) {
        return p + (q - p) * 6.0f * t;
      }
      if (t < 1.0f/2.0f) {
        return q;
      }
      if (t < 2.0f/3.0f) {
        return p + (q - p) * (2.0f/3.0f - t) * 6.0f;
      }
      return p;
    }
    static constexpr void hsl_to_rgb(f32_t h, f32_t s, f32_t l, f32_t& r, f32_t& g, f32_t& b) {
      if (s == 0.0f) {
        r = g = b = l;
        return;
      }
      f32_t q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
      f32_t p = 2.0f * l - q;
      r = hue_to_rgb(p, q, h + 1.0f/3.0f);
      g = hue_to_rgb(p, q, h);
      b = hue_to_rgb(p, q, h - 1.0f/3.0f);
    }
    static constexpr f32_t wrap01(f32_t v) {
      return v < 0.f ? v + 1.f : (v >= 1.f ? v - 1.f : v);
    }

    color hue(f32_t degrees) const;
    color saturation(f32_t amount) const;
    color lightness(f32_t amount) const;

    constexpr void set_hsl(f32_t hue, f32_t saturation, f32_t lightness) {
      r = hue;
      g = saturation;
      b = lightness;
      a = 1.0f;
    }
    static constexpr color hsl(f32_t hue, f32_t saturation, f32_t lightness) {
      fan::color c;
      c.set_hsl(hue, saturation, lightness);
      return c;
    }

  #if defined(FAN_GUI)
    constexpr color(const fan::graphics::gui::vec4_t& v) {
      r = v.x;
      g = v.y;
      b = v.z;
      a = v.w;
    }
    constexpr operator fan::graphics::gui::vec4_t() const {
      return *(fan::graphics::gui::vec4_t*)this;
    }
    constexpr operator fan::vec3() const {
      return fan::vec3{ r, g, b };
    }
    fan::graphics::gui::u32_t get_gui_color() const {
      return get_abgr();
    }
  #endif

    using value_type = f32_t;

    constexpr operator fan::vec4_wrap_t<value_type>() {
      return *(fan::vec4_wrap_t<value_type>*)this;
    }

    static fan::color hsv(f32_t H, f32_t S, f32_t V);
    static constexpr color rgb(cf_t r, cf_t g, cf_t b, cf_t a = 255) {
      return color(r / 255.f, g / 255.f, b / 255.f, a / 255.f);
    }

    static constexpr color readable_text(const color& background) {
      f32_t luminance = 0.2126f * background.r + 0.7152f * background.g + 0.0722f * background.b;
      if (luminance > 0.5f) {
        return color(0, 0, 0, 1);
      }
      else {
        return color(1, 1, 1, 1);
      }
    }


    static constexpr uint32_t size() {
      return 4;
    }
    void randomize();

    std::string to_string() const noexcept;
    void from_string(const std::string& str);
    static color parse(const std::string& str);
    friend std::ostream& operator<<(std::ostream& os, const color& c) noexcept;

    constexpr auto begin() const {
      return &r;
    }
    constexpr auto begin() {
      return &r;
    }
    constexpr auto end() const {
      return begin() + size();
    }
    constexpr auto end() {
      return begin() + size();
    }
    constexpr color set_alpha(f32_t alpha) const {
      return color(r, g, b, alpha);
    }

    constexpr f32_t get_brightest_channel() const {
      f32_t max_channel = std::max({ r, g, b });
      return max_channel;
    }
    constexpr f32_t get_brightness() const {
      return 0.299f * r + 0.587f * g + 0.114f * b;
    }

    constexpr color lerp(const color& other, f32_t t) const {
      return color(
        r + (other.r - r) * t,
        g + (other.g - g) * t,
        b + (other.b - b) * t,
        a + (other.a - a) * t
      );
    }
    f32_t r = 0, g = 0, b = 0, a = 1;
  };

  namespace colors {
    inline constexpr fan::color black = fan::color::from_rgba(0x000000FF);
    inline constexpr fan::color gray = fan::color::from_rgba(0x808080FF);
    inline constexpr fan::color red = fan::color::from_rgba(0xFF0000FF);
    inline constexpr fan::color green = fan::color::from_rgba(0x00FF00FF);
    inline constexpr fan::color blue = fan::color::from_rgba(0x0000FFFF);
    inline constexpr fan::color white = fan::color::from_rgba(0xFFFFFFFF);
    inline constexpr fan::color aqua = fan::color::from_rgba(0x00FFFFFF);
    inline constexpr fan::color purple = fan::color::from_rgba(0x800080FF);
    inline constexpr fan::color orange = fan::color::from_rgba(0xFFA500FF);
    inline constexpr fan::color pink = fan::color::from_rgba(0xFF35B8FF);
    inline constexpr fan::color yellow = fan::color::from_rgba(0xFFFF00FF);
    inline constexpr fan::color gold = fan::color::from_rgba(0xFFD700FF);
    inline constexpr fan::color cyan = fan::color::from_rgba(0x00FFFFFF);
    inline constexpr fan::color magenta = fan::color::from_rgba(0xFF00FFFF);
    inline constexpr fan::color transparent = fan::color::from_rgba(0x00000000);
    inline constexpr fan::color lime = fan::color(0.2f, 0.8f, 0.2f, 1.0f);
    inline constexpr fan::color brown = fan::color::from_rgb(0x8B6A3E);
  }
  namespace random {
    fan::color color();
    // always makes one channel brightest and scales other channels accordingly
    fan::color bright_color();
  }

  template <typename>
  inline constexpr bool is_color_type_v = false;

  template <>
  inline constexpr bool is_color_type_v<color> = true;

  template <typename T>
  concept is_color = is_color_type_v<std::remove_cvref_t<T>>;
}

#pragma pack(pop)