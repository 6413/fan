module;

#include <fan/physics/collision/triangle.h>

import fan.types.matrix;

export module fan.physics.collision.rectangle;

export namespace fan_2d {

	namespace collision {

		namespace rectangle {
      fan_api constexpr bool point_inside(const fan::vec2& p1, const fan::vec2& p2, const fan::vec2& p3, const fan::vec2& p4, const fan::vec2& point) {
				return fan_2d::collision::triangle::point_inside(p1, p2, p4, point) || fan_2d::collision::triangle::point_inside(p1, p3, p4, point);
			}

      // size is half
			fan_api constexpr bool point_inside_no_rotation(const fan::vec2& point, const fan::vec2& position, const fan::vec2& size) {
        return
          point.x >= position.x - size.x &&
          point.x <= position.x + size.x &&
          point.y >= position.y - size.y &&
          point.y <= position.y + size.y;
			}


      fan_api constexpr bool check_collision(const fan::vec2& center1, const fan::vec2& half_size1,
        const fan::vec2& center2, const fan::vec2& half_size2) {
        fan::vec2 abs_half_size1 = { fan::math::abs(half_size1.x), fan::math::abs(half_size1.y) };
        fan::vec2 abs_half_size2 = { fan::math::abs(half_size2.x), fan::math::abs(half_size2.y) };

        bool overlap_x = fan::math::abs(center1.x - center2.x) < (abs_half_size1.x + abs_half_size2.x);
        bool overlap_y = fan::math::abs(center1.y - center2.y) < (abs_half_size1.y + abs_half_size2.y);

        return overlap_x && overlap_y;
      }


      fan_api constexpr bool point_inside_rotated(const fan::vec2& point, const fan::vec2& position, const fan::vec2& size, const fan::vec3& angle, const fan::vec2& rotation_point) {
        // TODO BROKEN
        return 0;
        /*fan::mat4 m = fan::mat4(1);
        fan::mat4 t1 = fan::mat4(1).translate(-position - fan::vec3(rotation_point, 0));
        fan::mat4 t2 = fan::mat4(1).translate(position + fan::vec3(rotation_point, 0));
        fan::mat4 r = fan::mat4(1).rotate(-angle);
        m = t2 * r * t1;
        fan::vec4 rotated_point = m.inverse() * fan::vec4(fan::vec3(point), 1);

        return rotated_point.x >= position.x - size.x &&
          rotated_point.x <= position.x + size.x &&
          rotated_point.y >= position.y - size.y &&
          rotated_point.y <= position.y + size.y;*/
      }

			struct sides_e {
				static constexpr uint8_t top_left = 0;
				static constexpr uint8_t top_right = 1;
				static constexpr uint8_t bottom_right = 2;
				static constexpr uint8_t bottom_left = 3;
			};

			/* returns position of point
					 |-------|
			     | x | x |
					 |---|---|
					 | x | x |
			     |---|---|
			*/
      fan_api uint8_t get_side_collision(const fan::vec2& point, const fan::vec2& p, const fan::vec2& s) {
				if (point.x <= p.x && point.y <= p.y) {
					return sides_e::top_left;
				}
				if (point.x >= p.x && point.y <= p.y) {
					return sides_e::top_right;
				}
				if (point.x >= p.x && point.y >= p.y) {
					return sides_e::bottom_right;
				}
				return sides_e::bottom_left;
			}
		}
	}
}