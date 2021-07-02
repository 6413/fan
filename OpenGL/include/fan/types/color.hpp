#pragma once
#include <iostream>

#include <fan/types/types.hpp>

namespace fan {

	// defaultly gets values in format 0-1f, optional functions fan::color::rgb, fan::color::hex
	class color {
	public:

		using value_type = cf_t;
	
		static constexpr color rgb(cf_t r, cf_t g, cf_t b, cf_t a = 255) {
			return color(r / 255.f, g / 255.f, b / 255.f, a / 255.f);
		}

		static constexpr color hex(unsigned int hex) {
			if (hex <= 0xffffff) {
				return color::rgb(
					(hex >> 16) & 0xff,
					(hex >> 8) & 0xff,
					(hex >> 0) & 0xff
				);
			}
			return color::rgb(
				(hex >> 24) & 0xff,
				(hex >> 16) & 0xff,
				(hex >> 8) & 0xff,
				(hex >> 0) & 0xff
			);
		}

		cf_t r, g, b, a;
		constexpr color() : r(0), g(0), b(0), a(1) {}

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
			this->a = 1;
		}
		color& operator&=(const color& color_) {
			color ret;
			ret.r = (unsigned int)r & (unsigned int)color_.r;
			ret.g = (unsigned int)g & (unsigned int)color_.g;
			ret.b = (unsigned int)b & (unsigned int)color_.b;
			ret.a = (unsigned int)a & (unsigned int)color_.a;
			return *this;
		}
		color operator^=(const color& color_) {
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
		constexpr cf_t operator[](size_t x) const {
			return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
		}
		constexpr color operator-=(const color& color_) {
			return color(r -= color_.r, g -= color_.g, b -= color_.b, a -= color_.a);
		}
		constexpr color operator-(const color& color_) const {
			return color(r - color_.r, g - color_.g, b - color_.b, color_.a != 1 ? a - color_.a : a);
		}
		constexpr color operator+(const color& color_) const {
			return color(r + color_.r, g + color_.g, b + color_.b, a + color_.a);
		}
		template <typename T>
		constexpr color operator*(T value) const {
			return color(r * value, g * value, b * value);
		}
		void print() const {
			std::cout << r << " " << g << " " << b << " " << a << std::endl;
		}
		auto data() const {
			return &r;
		}

		static constexpr auto size() {
			return 4;
		}

	};

	namespace colors {

		static constexpr fan::color black =  fan::color(0, 0, 0);
		static constexpr fan::color red = fan::color(1, 0, 0);
		static constexpr fan::color green = fan::color(0, 1, 0);
		static constexpr fan::color blue = fan::color(0, 0, 1);
		static constexpr fan::color white = fan::color(1, 1, 1);
		static constexpr fan::color aqua = fan::color::hex(0x00FFFF);
		static constexpr fan::color purple = fan::color::hex(0x800080);
		static constexpr fan::color orange = fan::color::hex(0xFFA500);
		static constexpr fan::color pink = fan::color::hex(0xFFC0CB);
		static constexpr fan::color yellow = fan::color::hex(0xFFFF00);
		static constexpr fan::color cyan = fan::color::hex(0x00FFFF);
		static constexpr fan::color magenta = fan::color::hex(0xFF00FF);

	}

	inline std::ostream& operator<<(std::ostream& os, const color& color_) noexcept
	{
		os << color_.r << ' ';
		os << color_.g << ' ';
		os << color_.b << ' ';
		os << color_.a << '\n';
		return os;
	}
}