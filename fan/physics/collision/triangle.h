#pragma once

#include _FAN_PATH(types/vector.h)
#include _FAN_PATH(math/math.h)

namespace fan_2d {

	namespace collision {

		namespace triangle {


			constexpr bool point_inside(const fan::vec2& v1, const fan::vec2& v2, const fan::vec2& v3, const fan::vec2& point) {
				float d1 = 0, d2 = 0, d3 = 0;
				bool has_neg = 0, has_pos = 0;

				d1 = fan::math::sign(point, v1, v2);
				d2 = fan::math::sign(point, v2, v3);
				d3 = fan::math::sign(point, v3, v1);

				has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
				has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

				return !(has_neg && has_pos);
			}

		}

	}

}