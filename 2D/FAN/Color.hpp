#pragma once

class Color {
public:
	float r, g, b, a;
	Color() : r(0), g(0), b(0), a(0) {}

	Color(float r, float g, float b, float a);
};