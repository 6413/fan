#pragma once


#include <fan/types/vector.h>

#if defined(loco_imgui)
#include <fan/imgui/imgui.h>
#endif

#pragma pack(push, 1)

namespace fan {

	// defaultly gets values in format 0-1f, optional functions fan::color::rgb, fan::color::hex
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

#if defined(loco_imgui)
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
      return fan::vec3{r, g, b};
    }
#endif
    constexpr uint32_t to_u32() const {
		return static_cast<uint32_t>((static_cast<uint32_t>(std::clamp(r, 0.0f, 1.0f) * 255) << 0) |
        (static_cast<uint32_t>(std::clamp(g, 0.0f, 1.0f) * 255) << 8) |
        (static_cast<uint32_t>(std::clamp(b, 0.0f, 1.0f) * 255) << 16) |
        (static_cast<uint32_t>(std::clamp(a, 0.0f, 1.0f) * 255) << 24));
    }


    template <typename T>
    constexpr operator fan::vec4_wrap_t<T>() {
      return *(fan::vec4_wrap_t<T>*)this;
    }

		using value_type = cf_t;

		// returns rgb from hsv
    static fan::color hsv(f32_t H, f32_t S, f32_t V);
	
		static constexpr color rgb(cf_t r, cf_t g, cf_t b, cf_t a = 255) {
			return color(r / 255.f, g / 255.f, b / 255.f, a / 255.f);
		}

		static constexpr color hex(unsigned int hex) {
			return color::rgb(
				(hex >> 24) & 0xff,
				(hex >> 16) & 0xff,
				(hex >> 8) & 0xff,
				(hex >> 0) & 0xff
			);
		}
    // reverse endian
    static constexpr color hexr(unsigned int hex) {
			return color::rgb(
        (hex >> 0) & 0xff,
        (hex >> 8) & 0xff,
        (hex >> 16) & 0xff,
        (hex >> 24) & 0xff
      );
		}
    static constexpr color hexa(unsigned int hex) {
			return color::rgb(
        (hex >> 16) & 0xff,
        (hex >> 8) & 0xff,
        (hex >> 0) & 0xff,
        0xff
      );
		}
    static constexpr color hexar(unsigned int hex) {
			return color::rgb(
        (hex >> 0) & 0xff,
        (hex >> 8) & 0xff,
        (hex >> 16) & 0xff,
        0xff
      );
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
    cf_t* data();

    constexpr uint32_t get_hex() const {
      return ((uint32_t)(r * 255.0) << 24) | ((uint32_t)(g * 255.0) << 16) | ((uint32_t)(b * 255.0) << 8) | (uint32_t)(a * 255.0);
    }
    constexpr uint32_t get_hexr() const {
      return ((uint32_t)(r * 255.0)) | ((uint32_t)(g * 255.0) << 8) | ((uint32_t)(b * 255.0) << 16) | ((uint32_t)((a * 255.0))<<8) ;
    }

		static constexpr uint32_t size() {
			return 4;
		}

    private:
      static constexpr auto float_accuracy = 1000000;

      int64_t value_i64(int64_t min, int64_t max);

      f32_t value_f32(f32_t min, f32_t max);
    public:

    void randomize();

		std::string to_string() const noexcept;
		void from_string(const std::string& str);
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

		static constexpr fan::color black =  fan::color(0, 0, 0);
		static constexpr fan::color gray = fan::color::hex(0x808080FF);
		static constexpr fan::color red = fan::color(1, 0, 0);
		static constexpr fan::color green = fan::color(0, 1, 0);
		static constexpr fan::color blue = fan::color(0, 0, 1);
		static constexpr fan::color white = fan::color(1, 1, 1);
		static constexpr fan::color aqua = fan::color::hex(0x00FFFFFF);
		static constexpr fan::color purple = fan::color::hex(0x800080FF);
		static constexpr fan::color orange = fan::color::hex(0xFFA500FF);
		static constexpr fan::color pink = fan::color::hex(0xFF35B8FF);
		static constexpr fan::color yellow = fan::color::hex(0xFFFF00FF);
		static constexpr fan::color cyan = fan::color::hex(0x00FFFFFF);
		static constexpr fan::color magenta = fan::color::hex(0xFF00FFFF);
		static constexpr fan::color transparent = fan::color(0, 0, 0, 0);
	}
}

#pragma pack(pop)