#pragma once
#include <FAN/Math.hpp>
#include <FAN/Graphics.hpp>
#include <map>


struct LightDirection {
	Vec2 left;
	Vec2 right;
	Vec2 up;
	Vec2 down;
	LightDirection() : left(0), right(0), up(0), down(0) {}
	LightDirection(const float lightLen, const Vec2& start) {
		left = Vec2(-lightLen + start.x, start.y); right = Vec2(lightLen + start.x, start.y);
		up = Vec2(start.x, lightLen + start.y); down = Vec2(start.x, -lightLen + start.y);
	}
};

template <typename _Ty>
bool Trace(const __Vec2<_Ty> start, const float angle, const __Vec2<_Ty> pStart, const __Vec2<_Ty> pEnd) {
	Vec2 inter = IntersectionPoint(start, start + Vec2(1000, 0), pStart, pEnd);
	if (inter.x != -1 ) {
		return true;
	}
	return false;
}