#pragma once

class Color {
public:
	float r, g, b, a;
	Color() : r(0), g(0), b(0), a(0) {}
	
	Color(float r, float g, float b, float a);

	constexpr float operator[](size_t x) const {
		return !x ? this->r : x == 1 ? this->g : x == 2 ? this->b : x == 3 ? this->a : this->a;
	}
};

template <typename color_t, typename _Type>
constexpr color_t operator/(const color_t& c, _Type value) {
	return color_t(c.r / value, c.g / value, c.b / value, c.a / value);
}