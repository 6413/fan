#pragma once
#include <iostream>

#include _FAN_PATH(types/types.h)

#include <cmath>

namespace fan {

	// defaultly gets values in format 0-1f, optional functions fan::color::rgb, fan::color::hex
	class color {
	public:

    color(const fan::vec4& v) {
      *(fan::vec4*)this = v;
    }

		using value_type = cf_t;

		// returns rgb from hsv
		static fan::color hsv(f32_t H, f32_t S,f32_t V){

			f32_t s = S/100;
			f32_t v = V/100;
			f32_t C = s*v;
			f32_t X = C*(1-std::abs(fmod(H/60.0, 2)-1));
			f32_t m = v-C;
			f32_t r,g,b;
			if(H >= 0 && H < 60){
				r = C,g = X,b = 0;
			}
			else if(H >= 60 && H < 120){
				r = X,g = C,b = 0;
			}
			else if(H >= 120 && H < 180){
				r = 0,g = C,b = X;
			}
			else if(H >= 180 && H < 240){
				r = 0,g = X,b = C;
			}
			else if(H >= 240 && H < 300){
				r = X,g = 0,b = C;
			}
			else{
				r = C,g = 0,b = X;
			}
			int R = (r+m)*255;
			int G = (g+m)*255;
			int B = (b+m)*255;

			return fan::color::rgb(R, G, B, 255);
		}
	
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
			this->a = value;
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
		void print() const {
			std::cout << "{ " << r << ", " << g << ", " << b << ", " << a << " }";
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

	inline std::ostream& operator<<(std::ostream& os, const color& color_) noexcept
	{
		os << "{ ";
		os << color_.r << ", ";
		os << color_.g << ", ";
		os << color_.b << ", ";
		os << color_.a << " }";

		return os;
	}
}