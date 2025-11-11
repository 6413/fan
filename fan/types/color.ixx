module;

//#undef fan_gui

#if defined(fan_gui)
	#include <fan/imgui/imgui.h>
#endif

#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

export module fan.types.color;

import fan.types;
import fan.types.vector;
import fan.random;

#pragma pack(push, 1)

export namespace fan {
	// internal format 0-1
  template <typename type_t>
	struct color_ {
		constexpr color_() = default;
		template <typename T>
		constexpr color_(const fan::vec3_wrap_t<T>& v) {
			r = v.x;
			g = v.y;
			b = v.z;
		}
		template <typename T>
		constexpr color_(const fan::vec4_wrap_t<T>& v) {
			*(fan::vec4*)this = v;
		}
		constexpr color_(cf_t r, cf_t g, cf_t b, cf_t a = 1) : r(r), g(g), b(b), a(a) {
			this->r = r;
			this->g = g;
			this->b = b;
			this->a = a;
		}
		constexpr color_(cf_t value) : r(0), g(0), b(0), a(0) {
			this->r = value;
			this->g = value;
			this->b = value;
			this->a = value;
		}
		constexpr color_& operator&=(const color_& c) {
			r = (unsigned int)r & (unsigned int)c.r;
			g = (unsigned int)g & (unsigned int)c.g;
			b = (unsigned int)b & (unsigned int)c.b;
			a = (unsigned int)a & (unsigned int)c.a;
			return *this;
		}
		constexpr color_& operator^=(const color_& c) {
			r = (int)r ^ (int)c.r;
			g = (int)g ^ (int)c.g;
			b = (int)b ^ (int)c.b;
			return *this;
		}
		constexpr bool operator!=(const color_& c) const {
			return r != c.r || g != c.g || b != c.b;
		}
		constexpr bool operator==(const color_& c) const {
			return r == c.r && g == c.g && b == c.b && a == c.a;
		}
		constexpr cf_t& operator[](size_t x) {
			return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
		}
		constexpr cf_t operator[](size_t x) const {
			return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
		}
		constexpr color_ operator-=(const color_& c) {
			return color_(r -= c.r, g -= c.g, b -= c.b, a -= c.a);
		}
		constexpr color_ operator-() const {
			return color_(-r, -g, -b, a);
		}
		constexpr color_ operator-(const color_& c) const {
			return color_(r - c.r, g - c.g, b - c.b, c.a != 1 ? a - c.a : a);
		}
		constexpr color_ operator+(const color_& c) const {
			return color_(r + c.r, g + c.g, b + c.b, a + c.a);
		}
		template <typename T>
		constexpr color_ operator*(T value) const {
			return color_(r * value, g * value, b * value, a * value);
		}
		template <typename T>
		constexpr color_ operator/(T value) const {
			return color_(r / value, g / value, b / value);
		}

		template <typename T>
		constexpr color_ mult_no_alpha(T value) const {
			return color_(r * value, g * value, b * value);
		}
		cf_t* data() {
			return &r;
		}

    static constexpr f32_t clamp(f32_t value, f32_t min, f32_t max) {
      return value < min ? min : (value > max ? max : value);
    }

		static constexpr uint32_t to_byte(f32_t value) {
			return static_cast<uint32_t>(color_::clamp(value, 0.0f, 1.0f) * 255);
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

		static constexpr fan::color_<type_t> from_rgba(uint32_t color_) {
			fan::color_<type_t> c;
			c.set_rgba(color_);
			return c;
		}
		static constexpr fan::color_<type_t> from_abgr(uint32_t color_) {
			fan::color_<type_t> c;
			c.set_abgr(color_);
			return c;
		}
		static constexpr fan::color_<type_t> from_argb(uint32_t color_) {
			fan::color_<type_t> c;
			c.set_argb(color_);
			return c;
		}
		static constexpr fan::color_<type_t> from_bgra(uint32_t color_) {
			fan::color_<type_t> c;
			c.set_bgra(color_);
			return c;
		}

		static constexpr fan::color_<type_t> from_rgb(uint32_t color_) {
			fan::color_<type_t> c;
			uint32_t rgba_color = (color_ << 8) | 0xFF;
			c.set_rgba(rgba_color);
			return c;
		}

	#if defined(fan_gui)
		constexpr color_(const ImVec4& v) {
			r = v.x;
			g = v.y;
			b = v.z;
			a = v.w;
		}
		constexpr operator ImVec4() const {
			return *(ImVec4*)this;
		}
		constexpr operator fan::vec3() const {
			return fan::vec3{ r, g, b };
		}
		ImU32 get_imgui_color() const {
			return get_abgr();
		}
	#endif

		using value_type = type_t;

		constexpr operator fan::vec4_wrap_t<value_type>() {
			return *(fan::vec4_wrap_t<value_type>*)this;
		}


		// returns rgb from hsv
		static fan::color_<type_t> hsv(f32_t H, f32_t S, f32_t V) {
			f32_t s = S / 100, v = V / 100;
			f32_t C = s * v;
			f32_t X = C * (1 - std::abs(fmod(H / 60.0, 2) - 1));
			f32_t m = v - C;

			int i = static_cast<int>(H / 60) % 6;
			f32_t rgb[6][3] = { {C,X,0},{X,C,0},{0,C,X},{0,X,C},{X,0,C},{C,0,X} };

			return fan::color_<type_t>(rgb[i][0] + m, rgb[i][1] + m, rgb[i][2] + m, 1.0f);
		}

		static constexpr color_<type_t> rgb(cf_t r, cf_t g, cf_t b, cf_t a = 255) {
			return color_<type_t>(r / 255.f, g / 255.f, b / 255.f, a / 255.f);
		}

		static constexpr color_<type_t> readable_text(const color_<type_t>& background) {
			f32_t luminance = 0.2126f * background.r + 0.7152f * background.g + 0.0722f * background.b;
			if (luminance > 0.5f) {
				return color_(0, 0, 0, 1);
			}
			else {
				return color_<type_t>(1, 1, 1, 1);
			}
		}


		static constexpr uint32_t size() {
			return 4;
		}


		void randomize() {
			*this = fan::color_<type_t>(
				fan::random::value(0.f, 1.f),
				fan::random::value(0.f, 1.f),
				fan::random::value(0.f, 1.f),
				1
			);
		}

		std::string to_string() const noexcept {
      return "{ " +
        std::to_string(r) + ", " +
        std::to_string(g) + ", " +
        std::to_string(b) + ", " +
        std::to_string(a) + " }";
    }
    void from_string(const std::string& str) {
      std::string s;
      // remove braces and spaces manually
      for (char c : str) {
        if (c != '{' && c != '}' && c != ' ') {
          s += c;
        }
      }

      std::stringstream ss(s);
      std::string item;
      value_type values[4] = { 0, 0, 0, 1 };
      size_t i = 0;

      while (std::getline(ss, item, ',') && i < 4) {
        try {
          values[i++] = std::stof(item);
        }
        catch (...) {
          values[i++] = value_type(); // fallback
        }
      }

      r = values[0];
      g = values[1];
      b = values[2];
      a = values[3];
    }
    static color_ parse(const std::string& str) {
      color_ out;
      out.from_string(str);
      return out;
    }

		friend std::ostream& operator<<(std::ostream& os, const color_& c) noexcept {
			os << c.to_string();

			return os;
		}

		auto begin() const {
			return &r;
		}
		auto begin() {
			return &r;
		}
		auto end() const {
			return begin() + size();
		}
		auto end() {
			return begin() + size();
		}
		color_ set_alpha(f32_t alpha) const {
			return color_(r, g, b, alpha);
		}

    f32_t get_brightest_channel() const {
			f32_t max_channel = std::max({ r, g, b });
			return max_channel;
		}
    constexpr f32_t get_brightness() const {
      return 0.299f * r + 0.587f * g + 0.114f * b;
    }

    constexpr color_ lerp(const color_& other, f32_t t) const {
      return color_(
        r + (other.r - r) * t,
        g + (other.g - g) * t,
        b + (other.b - b) * t,
        a + (other.a - a) * t
      );
    }


		value_type r = 0, g = 0, b = 0, a = 1;
	};

  using color = color_<cf_t>;

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
	}
	namespace random {
		fan::color color() {
			return fan::color(
				fan::random::value(0.f, 1.f),
				fan::random::value(0.f, 1.f),
				fan::random::value(0.f, 1.f),
				1
			);
		}
		// always makes one channel brightest and scales other channels accordingly
		fan::color bright_color() {
			fan::color rand_color = fan::random::color();
			f32_t max_channel = std::max({ rand_color.r, rand_color.g, rand_color.b });
			return rand_color / max_channel;
		}
	}

  template <typename>
  inline constexpr bool is_color_type_v = false;

  template <typename T>
  inline constexpr bool is_color_type_v<color_<T>> = true;

  template <typename T>
  concept is_color = is_color_type_v<std::remove_cvref_t<T>>;
}

#pragma pack(pop)