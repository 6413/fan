#pragma once
#include "Math.hpp"
#include "Texture.hpp"
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
	Vec2 inter = IntersectionPoint(start, Vec2(start.x + DirectionVector(angle).x, start.y),
		Vec2(pStart.x, pStart.y), Vec2(pEnd.x, pEnd.y));
	//	printf("%f %f\n", start.x, inter.x);
	if (inter.x != INFINITY && start.x < inter.x) {
		return true;
	}
	return false;
}