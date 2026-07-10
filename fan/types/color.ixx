module;

//#undef FAN_GUI

#include <fan/utility.h>

export module fan.types.color;

import std;

import fan.print.error;
import fan.types;
import fan.types.vector;

#if defined(FAN_GUI)
  import fan.graphics.gui.types;
#endif

#pragma pack(push, 1)

export namespace fan {
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
    constexpr color(const std::uint8_t* begin, const std::uint8_t* end) {
      const std::uint8_t* ptr = begin;
      int i = 0;
      while (ptr != end) {
        (*this)[i++] = *ptr / f32_t(255);
        ++ptr;
      }
    }
    constexpr color(f32_t value) : r(0), g(0), b(0), a(1) {
      this->r = value;
      this->g = value;
      this->b = value;
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
    constexpr f32_t& operator[](std::size_t x) {
      return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
    }
    constexpr f32_t operator[](std::size_t x) const {
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
    constexpr color operator*(const fan::vec3& value) const {
      return color(r * value.x, g * value.y, b * value.z, a);
    }
    constexpr color operator+(const fan::vec3& value) const {
      return color(r + value.x, g + value.y, b + value.z, a);
    }
    constexpr color operator-(const fan::vec3& value) const {
      return color(r - value.x, g - value.y, b - value.z, a);
    }
    constexpr color operator/(const fan::vec3& value) const {
      return color(r / value.x, g / value.y, b / value.z, a);
    }
    constexpr color& operator*=(const fan::vec3& value) {
      r *= value.x;
      g *= value.y;
      b *= value.z;
      return *this;
    }
    constexpr color& operator+=(const fan::vec3& value) {
      r += value.x;
      g += value.y;
      b += value.z;
      return *this;
    }
    constexpr color& operator-=(const fan::vec3& value) {
      r -= value.x;
      g -= value.y;
      b -= value.z;
      return *this;
    }
    constexpr color& operator/=(const fan::vec3& value) {
      r /= value.x;
      g /= value.y;
      b /= value.z;
      return *this;
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

    static constexpr std::uint32_t to_byte(f32_t value) {
      return static_cast<std::uint32_t>(color::clamp(value, 0.0f, 1.0f) * 255);
    }

    static constexpr std::uint32_t pack_color(f32_t c1, f32_t c2, f32_t c3, f32_t c4) {
      return (to_byte(c1) << 24) | (to_byte(c2) << 16) | (to_byte(c3) << 8) | to_byte(c4);
    }
    static constexpr void unpack_color(std::uint32_t color_, f32_t& c1, f32_t& c2, f32_t& c3, f32_t& c4) {
      c1 = ((color_ >> 24) & 0xFF) / 255.0f;
      c2 = ((color_ >> 16) & 0xFF) / 255.0f;
      c3 = ((color_ >> 8) & 0xFF) / 255.0f;
      c4 = (color_ & 0xFF) / 255.0f;
    }

    constexpr std::uint32_t get_rgba() const { return pack_color(r, g, b, a); }
    constexpr std::uint32_t get_abgr() const { return pack_color(a, b, g, r); }
    constexpr std::uint32_t get_argb() const { return pack_color(a, r, g, b); }
    constexpr std::uint32_t get_bgra() const { return pack_color(b, g, r, a); }

    constexpr void set_rgba(std::uint32_t color_) { unpack_color(color_, r, g, b, a); }
    constexpr void set_abgr(std::uint32_t color_) { unpack_color(color_, a, b, g, r); }
    constexpr void set_argb(std::uint32_t color_) { unpack_color(color_, a, r, g, b); }
    constexpr void set_bgra(std::uint32_t color_) { unpack_color(color_, b, g, r, a); }

    constexpr std::uint16_t get_rgb565() const {
      std::uint32_t r5 = std::uint32_t(color::clamp(r, 0.f, 1.f) * 31.f + 0.5f);
      std::uint32_t g6 = std::uint32_t(color::clamp(g, 0.f, 1.f) * 63.f + 0.5f);
      std::uint32_t b5 = std::uint32_t(color::clamp(b, 0.f, 1.f) * 31.f + 0.5f);
      return std::uint16_t((r5 << 11) | (g6 << 5) | b5);
    }
    constexpr void set_rgb565(std::uint16_t v) {
      r = ((v >> 11) & 0x1F) / 31.f;
      g = ((v >> 5) & 0x3F) / 63.f;
      b = (v & 0x1F) / 31.f;
      a = 1.0f;
    }
    static constexpr fan::color from_rgba(std::uint32_t color_) {
      fan::color c;
      c.set_rgba(color_);
      return c;
    }
    static constexpr fan::color from_abgr(std::uint32_t color_) {
      fan::color c;
      c.set_abgr(color_);
      return c;
    }
    static constexpr fan::color from_argb(std::uint32_t color_) {
      fan::color c;
      c.set_argb(color_);
      return c;
    }
    static constexpr fan::color from_bgra(std::uint32_t color_) {
      fan::color c;
      c.set_bgra(color_);
      return c;
    }

    static constexpr fan::color from_rgb(std::uint32_t color_) {
      fan::color c;
      std::uint32_t rgba_color = (color_ << 8) | 0xFF;
      c.set_rgba(rgba_color);
      return c;
    }

    static constexpr fan::color from_rgb565(std::uint16_t v) {
      fan::color c;
      c.set_rgb565(v);
      return c;
    }


    static f32_t srgb_to_linear_channel(f32_t c);
    static f32_t linear_to_srgb_channel(f32_t c);
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
    color(const fan::graphics::gui::vec4_t& v);
    operator fan::graphics::gui::vec4_t() const;
  #endif
    constexpr operator fan::vec3() const {
      return fan::vec3{ r, g, b };
    }
    constexpr operator fan::vec4() const {
      return fan::vec4{ r, g, b, a };
    }
    constexpr fan::graphics::gui::u32_t get_gui_color() const {
      return get_abgr();
    }
    constexpr operator fan::graphics::gui::u32_t() const {
      return get_gui_color();
    }

    using value_type = f32_t;

    static fan::color hsv(f32_t H, f32_t S, f32_t V);
    static fan::vec3 to_hsv(const fan::color& c);
    fan::color shift_hue(f32_t degrees) const;

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

    static constexpr std::uint32_t size() {
      return 4;
    }
    void randomize();

    std::string to_string() const noexcept;
    friend std::ostream& operator<<(std::ostream& os, const color& c) {
      return os << c.to_string();
    }
    void from_string(const std::string& str);
    static color parse(const std::string& str);

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
      return r > g ? (r > b ? r : b) : (g > b ? g : b);
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
    constexpr color mix(const color& other, f32_t t) const {
      return this->lerp(other, t);
    }

    static constexpr fan::color nibble(std::uint8_t byte) {
      if (byte == 0x00)                 return fan::color(0.4f, 0.4f, 0.4f, 1.f);
      if (byte == 0xFF)                 return fan::color(1.0f, 1.0f, 1.0f, 1.f);
      if (byte == 0x0A || byte == 0x0D) return fan::color(1.0f, 0.0f, 0.0f, 1.f);
      int n = (byte >> 4) & 0xF;
      f32_t h = n / 16.f; // 0..1
      f32_t r, g, b;
      hsl_to_rgb(h, 0.75f, 0.65f, r, g, b);
      return fan::color(r, g, b, 1.f);
    }

    f32_t r = 0, g = 0, b = 0, a = 1;
  };

  constexpr std::uint16_t pack_rgb565(const fan::vec3& color) {
    fan::vec3 c = color.clamp(0.f, 1.f);
    std::uint32_t r = std::uint32_t(c.x * 31.f + 0.5f);
    std::uint32_t g = std::uint32_t(c.y * 63.f + 0.5f);
    std::uint32_t b = std::uint32_t(c.z * 31.f + 0.5f);
    return std::uint16_t((r << 11) | (g << 5) | b);
  }
  
  constexpr std::uint32_t _fan_check_24bit(unsigned long long v) {
    if (v > 0xFFFFFF) fan::throw_error_impl("literal must be 24-bit (0xRRGGBB)");
    return static_cast<std::uint32_t>(v);
  }
  constexpr std::uint32_t _fan_check_32bit(unsigned long long v) {
    if (v > 0xFFFFFFFF) fan::throw_error_impl("literal must be 32-bit (0xAARRGGBB etc.)");
    return static_cast<std::uint32_t>(v);
  }

  namespace color_literals {
    constexpr fan::color operator""_rgb(unsigned long long v) {
      return fan::color::from_rgb(_fan_check_24bit(v));
    }
    constexpr fan::color operator""_rgba(unsigned long long v) {
      return fan::color::from_rgba(_fan_check_32bit(v));
    }
    constexpr fan::color operator""_argb(unsigned long long v) {
      return fan::color::from_argb(_fan_check_32bit(v));
    }
    constexpr fan::color operator""_abgr(unsigned long long v) {
      return fan::color::from_abgr(_fan_check_32bit(v));
    }
    constexpr fan::color operator""_bgra(unsigned long long v) {
      return fan::color::from_bgra(_fan_check_32bit(v));
    }
    constexpr fan::color operator""_gray(unsigned long long v) {
      if (v > 255) fan::throw_error_impl("gray literal must be 0–255");
      return fan::color::rgb(v, v, v);
    }

    fan::color operator""_hsl(const char* str, std::size_t len) {
      f32_t h = 0, s = 0, l = 0;
      std::sscanf(str, "%f,%f,%f", &h, &s, &l);
      return fan::color::hsl(h / 360.f, s / 100.f, l / 100.f);
    }
    fan::color operator""_hsv(const char* str, std::size_t len) {
      f32_t h = 0, s = 0, v = 0;
      std::sscanf(str, "%f,%f,%f", &h, &s, &v);
      return fan::color::hsv(h / 360.f, s / 100.f, v / 100.f);
    }
  }

  using namespace fan::color_literals;

  namespace colors {
    inline constexpr fan::color amber = 0xFFBF00FF_rgba;
    inline constexpr fan::color aqua = 0x00FFFFFF_rgba;
    inline constexpr fan::color black = 0x000000FF_rgba;
    inline constexpr fan::color blue = 0x0000FFFF_rgba;
    inline constexpr fan::color brown = 0x8B6A3EFF_rgba;
    inline constexpr fan::color coral = 0xFF7F50FF_rgba;
    inline constexpr fan::color crimson = 0xDC143CFF_rgba;
    inline constexpr fan::color cyan = 0x00FFFFFF_rgba;
    inline constexpr fan::color dark_blue = 0x00008BFF_rgba;
    inline constexpr fan::color dark_cyan = 0x008B8BFF_rgba;
    inline constexpr fan::color dark_gray = 0x505050FF_rgba;
    inline constexpr fan::color dark_green = 0x006400FF_rgba;
    inline constexpr fan::color dark_orange = 0xFF8C00FF_rgba;
    inline constexpr fan::color dark_red = 0x8B0000FF_rgba;
    inline constexpr fan::color gold = 0xFFD700FF_rgba;
    inline constexpr fan::color gray = 0x808080FF_rgba;
    inline constexpr fan::color green = 0x00FF00FF_rgba;
    inline constexpr fan::color indigo = 0x4B0082FF_rgba;
    inline constexpr fan::color ivory = 0xFFFFF0FF_rgba;
    inline constexpr fan::color khaki = 0xF0E68CFF_rgba;
    inline constexpr fan::color lavender = 0xE6E6FAFF_rgba;
    inline constexpr fan::color lime = 0x32CD32FF_rgba;
    inline constexpr fan::color magenta = 0xFF00FFFF_rgba;
    inline constexpr fan::color maroon = 0x800000FF_rgba;
    inline constexpr fan::color navy = 0x000080FF_rgba;
    inline constexpr fan::color olive = 0x808000FF_rgba;
    inline constexpr fan::color orange = 0xFFA500FF_rgba;
    inline constexpr fan::color pink = 0xFF35B8FF_rgba;
    inline constexpr fan::color plum = 0xDDA0DDFF_rgba;
    inline constexpr fan::color purple = 0x800080FF_rgba;
    inline constexpr fan::color red = 0xFF0000FF_rgba;
    inline constexpr fan::color salmon = 0xFA8072FF_rgba;
    inline constexpr fan::color silver = 0xC0C0C0FF_rgba;
    inline constexpr fan::color tan = 0xD2B48CFF_rgba;
    inline constexpr fan::color teal = 0x008080FF_rgba;
    inline constexpr fan::color transparent = 0x00000000_rgba;
    inline constexpr fan::color turquoise = 0x40E0D0FF_rgba;
    inline constexpr fan::color violet = 0xEE82EEFF_rgba;
    inline constexpr fan::color white = 0xFFFFFFFF_rgba;
    inline constexpr fan::color yellow = 0xFFFF00FF_rgba;
  }
  namespace random {
    fan::color color();
    fan::color bright_color();
    fan::color color_near(const fan::color& base, f32_t hue_variance = 30.f, f32_t sv_variance = 15.f);
  }

  template <typename>
  inline constexpr bool is_color_type_v = false;

  template <>
  inline constexpr bool is_color_type_v<color> = true;

  template <typename T>
  concept is_color = is_color_type_v<std::remove_cvref_t<T>>;

  void lerp_pixels(std::uint8_t* dst, const std::uint8_t* target, std::size_t size, f32_t t, std::uint8_t channels = 4);
}

#pragma pack(pop)
