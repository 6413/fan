module;

export module fan.physics.collision.circle;

import fan.types.vector;

export namespace fan_2d {
	namespace collision {
		namespace circle {

			constexpr bool point_inside(const fan::vec2& p, const fan::vec2& c, f64_t r) {
				double dx = c.x - p.x;
				double dy = c.y - p.y;
				dx *= dx;
				dy *= dy;
				double distance_squared = dx + dy;
				double radius_squared = r * r;
				return distance_squared <= radius_squared;
			}
      constexpr bool inside(const fan::vec2& c0, f64_t r0, const fan::vec2& c1, f64_t r1) {
        f64_t dx = c1.x - c0.x;
        f64_t dy = c1.y - c0.y;
        f64_t rs = r0 + r1;
        return dx * dx + dy * dy <= rs * rs;
      }
		}
	}
}