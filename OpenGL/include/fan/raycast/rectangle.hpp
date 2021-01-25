#pragma once

#include <fan/math/vector.hpp>

namespace fan_2d {

	namespace collision {

		namespace rectangle {

			constexpr bool point_inside(const fan::vec2& point, const fan::vec2& rectangle_position, const fan::vec2& rectangle_size);
			constexpr bool point_inside_boundary(const fan::vec2& point, const fan::vec2& rectangle_position, const fan::vec2& rectangle_size);

		}


	}

	namespace raycast {

		namespace rectangle {


			enum class sides {
				nothing,
				inside,
				left,
				right,
				up,
				down
			};

			namespace _private {

				enum class sideflags {
					left = 1,
					right = 2,
					up = 4,
					down = 8
				};

				constexpr uint8_t side_flag(const fan::vec2& position, const fan::vec2& wall_position, const fan::vec2& wall_size) {
					uint8_t flag = 0;
					if(position.x <= wall_position.x)
						flag |= (uint8_t)sideflags::left;
					if (position.x >= wall_position.x + wall_size.x)
						flag |= (uint8_t)sideflags::right;
					if (position.y <= wall_position.y)
						flag |= (uint8_t)sideflags::up;
					if (position.y >= wall_position.y + wall_size.y)
						flag |= (uint8_t)sideflags::down;
					return flag;
				}

				constexpr sides straight_side_giver(f_t angle, uint8_t flag, const fan::vec2& src, const fan::vec2& dst, const fan::vec2& wall_position, const fan::vec2& wall_size) {
					switch (flag) {
						case (uint8_t)sideflags::left:
						{
							if (!fan::aim_angle(src, wall_position + fan::vec2(0, wall_size.y))) {
								return sides::nothing;
							}
							else if (dst.y >= wall_position.y + wall_size.y) {
								if (fan::aim_angle(src, wall_position + fan::vec2(0, wall_size.y)) <= angle) {
									return sides::nothing;
								}
							}
							else if (dst.y <= wall_position.y){
								if (fan::aim_angle(src, wall_position) >= angle) {
									return sides::nothing;
								}
							}

							return (sides)((dst.x >= wall_position.x) * (uint8_t)sides::left);
						}
						case (uint8_t)sideflags::right:
						{

							if (dst.y <= wall_position.y){
								if (fan::aim_angle(src, wall_position + fan::vec2(wall_size.x, 0)) <= angle) {
									return sides::nothing;
								}
							}
							else if (dst.y >= wall_position.y + wall_size.y) {
								if (fan::aim_angle(src, wall_position + wall_size) >= angle) {
									return sides::nothing;
								}
							}
							return (sides)((dst.x <= wall_position.x + wall_size.x) * (uint8_t)sides::right);
						}
						case (uint8_t)sideflags::up:
						{
							
							if (std::round(src.y) == std::round(wall_position.y) &&
								std::round(fan::aim_angle(src, wall_position + fan::vec2(wall_size.x, 0))) == std::round(fan::pi / 2)) {
								return sides::nothing;
							}
							if (dst.x >= wall_position.x + wall_size.x) {
								if (fan::aim_angle(src, wall_position + fan::vec2(wall_size.x, 0)) >= angle) {
									return sides::nothing;
								}
							}
							else if (dst.x < wall_position.x){
								if (fan::aim_angle(src, wall_position) <= angle) {
									return sides::nothing;
								}
							}

							return (sides)((dst.y >= wall_position.y) * (uint8_t)sides::up);
						}
						case (uint8_t)sideflags::down:
						{
							if (dst.x <= wall_position.x){
								if (fan::aim_angle(src, wall_position + fan::vec2(0, wall_size.y)) > angle) {
									return sides::nothing;
								}
							}
							else if (dst.x >= wall_position.x + wall_size.x) {
								if (fan::aim_angle(src, wall_position + wall_size) <= angle) {
									return sides::nothing;
								}
							}
							return (sides)((dst.y <= wall_position.y + wall_size.y) * (uint8_t)sides::down);
						}
					}
					return sides::nothing; // could be wrong temp fix
				}

				constexpr sides side_giver(uint8_t flag, const fan::vec2& src, const fan::vec2& dst, const fan::vec2& wall_position, const fan::vec2& wall_size) {
					const auto angle = fan::aim_angle(src, dst);
					
					if (fan_2d::collision::rectangle::point_inside(src, wall_position, wall_size)) {
						return sides::inside;
					}
					if (dst.isnan()) {
						return sides::inside;
					}

					switch (flag) {
						case ((uint8_t)sideflags::left | (uint8_t)sideflags::up):
						{

							if (fan::aim_angle(src, wall_position) >= std::abs(angle)) {
								if (dst.x <= wall_position.x && dst.y >= wall_position.y) {
									return straight_side_giver(angle, (uint8_t)sideflags::left, src, dst, wall_position, wall_size);
								}
							}
							else if (fan::aim_angle(src, wall_position) >= angle) {
								if (dst.y > wall_position.y) {
									return straight_side_giver(angle, (uint8_t)sideflags::up, src, dst, wall_position, wall_size);
								}
							}

							return sides::nothing;
						}
						case ((uint8_t)sideflags::left | (uint8_t)sideflags::down):
						{
							if (fan::aim_angle(src, wall_position + fan::vec2(0, wall_size.y)) >= angle) {
								if (dst.x >= wall_position.x) {
									return straight_side_giver(angle, (uint8_t)sideflags::left, src, dst, wall_position, wall_size);
								}
							}
							else if (dst.y <= wall_position.y + wall_size.y && fan::aim_angle(src, wall_position + fan::vec2(0, wall_size.y)) <= angle) {
								return straight_side_giver(angle, (uint8_t)sideflags::down, src, dst, wall_position, wall_size);
							}
							
							return sides::nothing;
						}
						case ((uint8_t)sideflags::right | (uint8_t)sideflags::up):
						{

							if (fan::aim_angle(src, wall_position + fan::vec2(wall_size.x, 0)) >= std::abs(angle)) {
								if (dst.x <= wall_position.x + wall_size.x && dst.y >= wall_position.y) {
									return straight_side_giver(angle, (uint8_t)sideflags::right, src, dst, wall_position, wall_size);
								}
							}
							else if (dst.y > wall_position.y && fan::aim_angle(src, wall_position + fan::vec2(wall_size.x, 0)) <= angle) {
								return straight_side_giver(angle, (uint8_t)sideflags::up, src, dst, wall_position, wall_size);
							}
							return sides::nothing;
						}
						case ((uint8_t)sideflags::right | (uint8_t)sideflags::down):
						{
							if (fan::aim_angle(src, wall_position + wall_size) <= -std::abs(angle)) {
								if (dst.x <= wall_position.x + wall_size.x && dst.y <= wall_position.y + wall_size.y) {
									return straight_side_giver(angle, (uint8_t)sideflags::right, src, dst, wall_position, wall_size);
								}
							}
							else if (src.y == wall_position.y + wall_size.y) {
								return straight_side_giver(angle, (uint8_t)sideflags::right, src, dst, wall_position, wall_size);
							}
							else if (dst.y <= wall_position.y + wall_size.y) {
								return straight_side_giver(angle, (uint8_t)sideflags::down, src, dst, wall_position, wall_size);
							}
							return sides::nothing;
						}
						case (uint8_t)sideflags::left:
						{
							return straight_side_giver(angle, flag, src, dst, wall_position, wall_size);
						}
						case (uint8_t)sideflags::right:
						{
							return straight_side_giver(angle, flag, src, dst, wall_position, wall_size);
						}
						case (uint8_t)sideflags::up:
						{
							return straight_side_giver(angle, flag, src, dst, wall_position, wall_size);
						}
						case (uint8_t)sideflags::down:
						{
							return straight_side_giver(angle, flag, src, dst, wall_position, wall_size);
						}
					}
					return sides::nothing; // could be wrong temp fix
				}
			}


			struct raycast_t {
				fan::vec2 point;
				sides side;
			};

			// wall top left format
			constexpr raycast_t raycast(const fan::vec2& src, const fan::vec2& dst, const fan::vec2& wall_position, const fan::vec2& wall_size) {

				switch (_private::side_giver(_private::side_flag(src, wall_position, wall_size), src, dst, wall_position, wall_size)) {
					case sides::left:
					{
						const fan::vec2 velocity(dst - src);
						return raycast_t{ src + velocity * (((wall_position.x + wall_size.x * 0.5) - (src.x + wall_size.x * 0.5)) / velocity.x), sides::left };
					}
					case sides::right:
					{
						const fan::vec2 velocity(dst - src);
						return raycast_t{ src + velocity * (((wall_position.x + wall_size.x * 0.5) - (src.x - wall_size.x * 0.5)) / velocity.x), sides::right };
					}
					case sides::up:
					{
						const fan::vec2 velocity(dst - src);
						return raycast_t{ src + velocity * (((wall_position.y + wall_size.y * 0.5) - (src.y + wall_size.y * 0.5)) / velocity.y), sides::up };
					}
					case sides::down:
					{
						const fan::vec2 velocity(dst - src);
						return raycast_t{ src + velocity * (((wall_position.y + wall_size.y * 0.5) - (src.y - wall_size.y * 0.5)) / velocity.y), sides::down };
					}
					case sides::inside:
					{
						return raycast_t{ src, sides::inside };
					}
					default:
					{
						return raycast_t{ INFINITY, sides::nothing };
					}
				}
			}

		}
		
	}

}