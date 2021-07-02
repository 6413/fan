#pragma once

#include <fan/types/vector.hpp>

namespace fan_2d {

	namespace collision {

		namespace circle {

			constexpr bool point_inside(const fan::vec2& p, const fan::vec2& c, f_t r)
			{
				double dx = c.x - p.x;
				double dy = c.y - p.y;
				dx *= dx;
				dy *= dy;
				double distanceSquared = dx + dy;
				double radiusSquared = r * r;
				return distanceSquared <= radiusSquared;
			}

		}

	}

}