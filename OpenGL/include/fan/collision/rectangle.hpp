#pragma once
#include <fan/graphics/graphics.hpp>
#include <fan/raycast/rectangle.hpp>

namespace fan_2d {

	namespace collision {

		constexpr fan::da_t<uint_t, 2> get_velocity_corners_2d(const fan::da_t<f_t, 2>& vel) {
			if (vel[0] >= 0)
				if (vel[1] >= 0)
					return { 2, 1 };
				else
					return { 0, 3 };
			else
				if (vel[1] >= 0)
					return { 0, 3 };
				else
					return { 2, 1 };
		}

		constexpr fan::da_t<uint_t, 3> get_velocity_corners_3d(const fan::da_t<f32_t, 2>& vel) {
			if (vel[0] >= 0)
				if (vel[1] >= 0)
					return { 2, 1, 3 };
				else
					return { 0, 3, 1 };
			else
				if (vel[1] >= 0)
					return { 0, 3, 2 };
				else
					return { 2, 1, 0 };
		}

		namespace rectangle {

			constexpr bool collide(const fan::vec2& a, const fan::vec2& a_size, const fan::vec2& b, const fan::vec2& b_size) {
				return a[0] + a_size[0] > b[0] &&
					   a[0] < b[0] + b_size[0] &&
					   a[1] + a_size[1] > b[1] &&
					   a[1] < b[1] + b_size[1];
			}

			constexpr bool point_inside(const fan::vec2& point, const fan::vec2& rectangle_position, const fan::vec2& rectangle_size) {
				return point.x > rectangle_position.x && point.x < rectangle_position.x + rectangle_size.x &&
					   point.y > rectangle_position.y && point.y < rectangle_position.y + rectangle_size.y;
			}

			constexpr bool point_inside_boundary(const fan::vec2& point, const fan::vec2& rectangle_position, const fan::vec2& rectangle_size) {
				return point.x >= rectangle_position.x && point.x <= rectangle_position.x + rectangle_size.x &&
					   point.y >= rectangle_position.y && point.y <= rectangle_position.y + rectangle_size.y;
			}

			static void resolve_collision(fan_2d::rectangle& player, const fan_2d::rectangle& walls) {

				if (!player.get_velocity(0)) {
					return;
				}

				while (player.get_velocity(0) != 0) {
					const auto player_corners = player.get_corners();
					const auto player_points = fan_2d::collision::get_velocity_corners_3d(player.get_velocity(0));

					const auto player_position = player.get_position();
					const auto player_size = player.get_size();
					const auto player_velocity = player.get_velocity(0);

					fan_2d::raycast::rectangle::raycast_t closest;
					closest.point = fan::inf;

					uint_t closest_corner = fan::uninitialized;

					bool step1 = false;

					for (uint_t i = 0; i < walls.size(); ++i) {

						// step 1

					//	if (walls.get_size(i).x < player_size.x || walls.get_size(i).y < player_size.y) {
					//		const auto& wall_points = fan_2d::collision::GetPointsTowardsVelocity3(-player.get_velocity());
					//		const auto& wall_corners = walls.get_corners(i);
					//		for (const auto p : wall_points) {
					//			fan_2d::raycast::rectangle::raycast_t ray_point = fan_2d::raycast::rectangle::raycast(
					//				wall_corners[p],
					//				wall_corners[p] - player.get_velocity() * player.m_window.get_delta_time(),
					//				player_position, player_size);
					//			if (fan::ray_hit(ray_point.point)
					//			) 
					//			{
					//				auto closest_distance = fan_2d::distance(walls.get_position(i), closest.point);
					//				auto ray_point_distance = fan_2d::distance(walls.get_position(i), ray_point.point);

					//				if (ray_point_distance < closest_distance) {
					//					closest = ray_point;
					//					closest_corner = p;
					//					step1 = true;
					//					size = walls.get_size(i);
					//				}
					//			}
					//		}
					//	}
					//	else {
							for (const auto p : player_points) {
								fan_2d::raycast::rectangle::raycast_t ray_point = fan_2d::raycast::rectangle::raycast(
									player_corners[p],
									player_corners[p] + player.get_velocity(0) * player.m_camera->m_window->get_delta_time(),
									walls.get_position(i), walls.get_size(i));
								if (fan::ray_hit(ray_point.point)
								) 
								{
									auto closest_distance = fan_2d::distance(player_position, closest.point);
									auto ray_point_distance = fan_2d::distance(player_position, ray_point.point);

									if (ray_point_distance < closest_distance) {
										closest = ray_point;
										closest_corner = p;
										step1 = false;
									}
								}
							}
						}
					//}

					if (fan::ray_hit(closest.point)) {

					//	if (!step1) {
							if (closest_corner == 1) {
								closest.point[0] -= player_size[0];
							}
							if (closest_corner == 2) {
								closest.point[1] -= player_size[1];
							}
							if (closest_corner == 3) {
								closest.point[0] -= player_size[0];
								closest.point[1] -= player_size[1];
							}
					//	}
						/*else {
							if (closest_corner == 0) {
								closest.point[0] -= player_size[0];
								closest.point[1] -= player_size[1];
							}
							if (closest_corner == 1) {
								closest.point[0] -= player_size[0];
							}
							if (closest_corner == 2) {
								closest.point[1] -= player_size[1];
							}
							if (closest_corner == 3) {
								closest.point[0] -= player_size[0];
								closest.point[1] -= player_size[1];
							}
						}*/

						if (closest.side == fan_2d::raycast::rectangle::sides::inside)
						{
							player.set_position(0, player.get_position() - player_velocity * player.m_camera->m_window->get_delta_time());
							break;
						}

						player.set_position(0, closest.point);

						auto velocity = player.get_velocity(0);

						velocity[((closest.side == fan_2d::raycast::rectangle::sides::left || closest.side == fan_2d::raycast::rectangle::sides::right) && !player_velocity[0]) ||
							((closest.side == fan_2d::raycast::rectangle::sides::up || closest.side == fan_2d::raycast::rectangle::sides::down) && !!player_velocity[1])] = 0;

						player.set_velocity(0, velocity);
					}
					else {
						player.set_position(0, player.get_position() + player_velocity * player.m_camera->m_window->get_delta_time());
						break;
					}
				}

				//fan::print(ind);
			}

		}

	}

}
