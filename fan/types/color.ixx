module;

#if defined(fan_gui)
#include <fan/imgui/imgui.h>
#endif

#include <random>
#include <string>
#include <sstream>
#include <algorithm>

export module fan.types.color;

import fan.types.vector;

#pragma pack(push, 1)

export namespace fan {
	// internal format 0-1
	struct color {
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
    constexpr color() = default;
		constexpr color(cf_t r, cf_t g, cf_t b, cf_t a = 1) : r(r), g(g), b(b), a(a) {
			this->r = r;
			this->g = g;
			this->b = b;
			this->a = a;
		}
		constexpr color(cf_t value) : r(0), g(0), b(0), a(0) {
			this->r = value;
			this->g = value;
			this->b = value;
			this->a = value;
		}
		constexpr color& operator&=(const color& color_) {
			color ret;
			ret.r = (unsigned int)r & (unsigned int)color_.r;
			ret.g = (unsigned int)g & (unsigned int)color_.g;
			ret.b = (unsigned int)b & (unsigned int)color_.b;
			ret.a = (unsigned int)a & (unsigned int)color_.a;
			return *this;
		}
		constexpr color operator^=(const color& color_) {
			r = (int)r ^ (int)color_.r;
			g = (int)g ^ (int)color_.g;
			b = (int)b ^ (int)color_.b;
			return *this;
		}
		constexpr bool operator!=(const color& color_) const {
			return r != color_.r || g != color_.g || b != color_.b;
		}
		constexpr bool operator==(const color& color_) const {
			return r == color_.r && g == color_.g && b == color_.b && a == color_.a;
		}
		constexpr cf_t& operator[](size_t x) {
			return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
		}
    constexpr cf_t operator[](size_t x) const {
			return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
		}
		constexpr color operator-=(const color& color_) {
			return color(r -= color_.r, g -= color_.g, b -= color_.b, a -= color_.a);
		}
		constexpr color operator-() const {
			return color(-r, -g, -b, a);
		}
		constexpr color operator-(const color& color_) const {
			return color(r - color_.r, g - color_.g, b - color_.b, color_.a != 1 ? a - color_.a : a);
		}
		constexpr color operator+(const color& color_) const {
			return color(r + color_.r, g + color_.g, b + color_.b, a + color_.a);
		}
		template <typename T>
		constexpr color operator*(T value) const {
			return color(r * value, g * value, b * value, a * value);
		}
		template <typename T>
		constexpr color operator/(T value) const {
			return color(r / value, g / value, b / value);
		}

		template <typename T>
		constexpr color mult_no_alpha(T value) const {
			return color(r * value, g * value, b * value);
		}
    cf_t* data() {
      return &r;
    }

    static constexpr uint32_t to_byte(f32_t value) {
      return static_cast<uint32_t>(std::clamp(value, 0.0f, 1.0f) * 255);
    }

    static constexpr uint32_t pack_color(f32_t c1, f32_t c2, f32_t c3, f32_t c4) {
      return (to_byte(c1) << 24) | (to_byte(c2) << 16) | (to_byte(c3) << 8) | to_byte(c4);
    }
    static constexpr void unpack_color(uint32_t color, f32_t& c1, f32_t& c2, f32_t& c3, f32_t& c4) {
      c1 = ((color >> 24) & 0xFF) / 255.0f;
      c2 = ((color >> 16) & 0xFF) / 255.0f;
      c3 = ((color >> 8) & 0xFF) / 255.0f;
      c4 = (color & 0xFF) / 255.0f;
    }

    constexpr uint32_t get_rgba() const { return pack_color(r, g, b, a); }
    constexpr uint32_t get_abgr() const { return pack_color(a, b, g, r); }
    constexpr uint32_t get_argb() const { return pack_color(a, r, g, b); }
    constexpr uint32_t get_bgra() const { return pack_color(b, g, r, a); }

    constexpr void set_rgba(uint32_t color) { unpack_color(color, r, g, b, a); }
    constexpr void set_abgr(uint32_t color) { unpack_color(color, a, b, g, r); }
    constexpr void set_argb(uint32_t color) { unpack_color(color, a, r, g, b); }
    constexpr void set_bgra(uint32_t color) { unpack_color(color, b, g, r, a); }

    static constexpr fan::color from_rgba(uint32_t color) {
      fan::color c;
      c.set_rgba(color);
      return c;
    }
    static constexpr fan::color from_abgr(uint32_t color) {
      fan::color c;
      c.set_abgr(color);
      return c;
    }
    static constexpr fan::color from_argb(uint32_t color) {
      fan::color c;
      c.set_argb(color);
      return c;
    }
    static constexpr fan::color from_bgra(uint32_t color) {
      fan::color c;
      c.set_bgra(color);
      return c;
    }

  #if defined(fan_gui)
    constexpr color(const ImVec4& v) {
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

    template <typename T>
    constexpr operator fan::vec4_wrap_t<T>() {
      return *(fan::vec4_wrap_t<T>*)this;
    }

    using value_type = cf_t;

    // returns rgb from hsv
    static fan::color hsv(f32_t H, f32_t S, f32_t V) {
      f32_t s = S / 100, v = V / 100;
      f32_t C = s * v;
      f32_t X = C * (1 - std::abs(fmod(H / 60.0, 2) - 1));
      f32_t m = v - C;

      int i = static_cast<int>(H / 60) % 6;
      f32_t rgb[6][3] = { {C,X,0},{X,C,0},{0,C,X},{0,X,C},{X,0,C},{C,0,X} };

      return fan::color(rgb[i][0] + m, rgb[i][1] + m, rgb[i][2] + m, 1.0f);
    }

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

    private:
      static constexpr auto float_accuracy = 1000000;

    int64_t value_i64(int64_t min, int64_t max) {
      static std::random_device device;
      static std::mt19937_64 random(device());

      std::uniform_int_distribution<int64_t> distance(min, max);

      return distance(random);
    }


    f32_t value_f32(f32_t min, f32_t max) {
      return (f32_t)value_i64(min * float_accuracy, max * float_accuracy) / float_accuracy;
    }
    public:

    void randomize() {
      *this = fan::color(
        value_f32(0, 1),
        value_f32(0, 1),
        value_f32(0, 1),
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
      std::stringstream ss(str);
      char ch;
      for (size_t i = 0; i < 4; ++i) {
        ss >> ch >> (*this)[i];
      }
    }
		friend std::ostream& operator<<(std::ostream& os, const color& color_) noexcept {
  		os << color_.to_string();

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
    color set_alpha(f32_t alpha) const {
      return color(r, g, b, alpha);
    }

    cf_t r = 0, g = 0, b = 0, a = 1;
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
	}
}

#pragma pack(pop)