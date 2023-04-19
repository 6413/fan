#pragma once

#include _FAN_PATH(physics/collision/triangle.h)

namespace fan_2d {

	namespace collision {

		namespace rectangle {

			constexpr bool point_inside(const fan::vec2& p1, const fan::vec2& p2, const fan::vec2& p3, const fan::vec2& p4, const fan::vec2& point) {
				return fan_2d::collision::triangle::point_inside(p1, p2, p4, point) || fan_2d::collision::triangle::point_inside(p1, p3, p4, point);
			}

      // size is half
			constexpr bool point_inside_no_rotation(const fan::vec2& point, const fan::vec2& position, const fan::vec2& size) {
        return
          
         /* fan::math::abs(point.x - position.x) <= size.x &&
          fan::math::abs(point.y - position.y) <= size.y;*/
          
          point.x >= position.x - size.x &&
          point.x <= position.x + size.x &&
          point.y >= position.y - size.y &&
          point.y <= position.y + size.y;
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
			uint8_t get_side_collision(const fan::vec2& point, const fan::vec2& p, const fan::vec2& s) {
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

		//	constexpr fan::da_t<uintptr_t, 2> get_velocity_corners_2d(const fan::da_t<f_t, 2>& vel) {
		//		if (vel[0] >= 0)
		//			if (vel[1] >= 0)
		//				return { 2, 1 };
		//			else
		//				return { 0, 3 };
		//		else
		//			if (vel[1] >= 0)
		//				return { 0, 3 };
		//			else
		//				return { 2, 1 };
		//	}

		//	constexpr fan::da_t<uintptr_t, 3> get_velocity_corners_3d(const fan::da_t<f32_t, 2>& vel) {
		//		if (vel[0] >= 0)
		//			if (vel[1] >= 0)
		//				return { 2, 1, 3 };
		//			else
		//				return { 0, 3, 1 };
		//		else
		//			if (vel[1] >= 0)
		//				return { 0, 3, 2 };
		//			else
		//				return { 2, 1, 0 };
		//	}

		//	static void resolve_non_rotational_static_collision(fan_2d::opengl::rectangle& player, const fan_2d::opengl::rectangle& walls) {

		//		if (!player.get_velocity(0)) {
		//			return;
		//		}

		//		fan::vec2 original_velocity = player.get_velocity(0);

		//		while (player.get_velocity(0) != 0) {
		//			auto player_corners = player.get_corners();
		//			const auto player_points = fan_2d::collision::rectangle::get_velocity_corners_3d(player.get_velocity(0));

		//			const auto player_position = player.get_position();
		//			const auto player_size = player.get_size();
		//			const auto player_velocity = player.get_velocity(0);

		//			fan_2d::raycast::rectangle::raycast_t closest;
		//			closest.point = fan::math::inf;

		//			bool step1 = false;
		//			int index = 0;
		//			fan::vec2 vp = fan::uninitialized;

		//			f_t smallest_distance = fan::math::inf;

		//			for (uintptr_t i = 0; i < walls.size(); ++i) {

		//				// step 1

		//				//if (walls.get_size(i).x < player_size.x || walls.get_size(i).y < player_size.y) {
		//				for (const auto p : player_points) {
		//					fan_2d::raycast::rectangle::raycast_t ray_point = fan_2d::raycast::rectangle::raycast(
		//						player_corners[p],
		//						player_corners[p] + player.get_velocity(0) * player.m_camera->m_window->get_delta_time(),
		//						walls.get_position(i), walls.get_size(i));
		//					if (fan::math::ray_hit(ray_point.point)
		//						) 
		//					{
		//						auto ray_point_distance = fan_2d::math::distance(player_corners[p], ray_point.point);

		//						if (ray_point_distance < smallest_distance) {
		//							smallest_distance = ray_point_distance;
		//							closest = ray_point;
		//							step1 = false;
		//							index = i;
		//							vp = player_corners[p];
		//						}
		//					}
		//				}
		//				//	}
		//				//if (vp == fan::uninitialized) {
		//				const auto& wall_points = get_velocity_corners_3d(-player.get_velocity(0));
		//				const auto& wall_corners = walls.get_corners(i);
		//				for (const auto p : wall_points) {
		//					fan_2d::raycast::rectangle::raycast_t ray_point = fan_2d::raycast::rectangle::raycast(
		//						wall_corners[p],
		//						wall_corners[p] - player.get_velocity(0) * player.m_camera->m_window->get_delta_time(),
		//						player_position, player_size);
		//					if (fan::math::ray_hit(ray_point.point)
		//						) 
		//					{
		//						auto ray_point_distance = fan_2d::math::distance(wall_corners[p], ray_point.point);

		//						if (ray_point_distance < smallest_distance) {
		//							smallest_distance = ray_point_distance;
		//							closest = ray_point;
		//							step1 = true;
		//							index = i;
		//							vp = wall_corners[p];
		//						}
		//					}
		//				}	
		//				//}
		//			}

		//			if (fan::math::ray_hit(closest.point)) {

		//				if (closest.side == fan_2d::raycast::rectangle::sides::inside)
		//				{
		//					player.set_position(0, player.get_position() - original_velocity * player.m_camera->m_window->get_delta_time());
		//					break;
		//				}

		//				f_t smallest(fan::math::inf);

		//				for (const auto p : player_points) {
		//					smallest = std::min(smallest, fan_2d::math::distance(walls.get_corners(index)[p], closest.point));
		//				}

		//				//	if ((smallest == fan_2d::distance(vp, closest.point) && step1) || !step1) {
		//				if (step1) {
		//					player.set_position(0, player.get_position(0) - (closest.point - vp));
		//				}
		//				else {
		//					player.set_position(0, player.get_position(0) + (closest.point - vp));
		//				}
		//				//}

		//				auto velocity = player.get_velocity(0);

		//				velocity[((closest.side == fan_2d::raycast::rectangle::sides::left || closest.side == fan_2d::raycast::rectangle::sides::right) && !player_velocity[0]) ||
		//					((closest.side == fan_2d::raycast::rectangle::sides::up || closest.side == fan_2d::raycast::rectangle::sides::down) && !!player_velocity[1])] = 0;

		//				player.set_velocity(0, velocity);

		//			}
		//			else {
		//				player.set_position(0, player.get_position() + player_velocity * player.m_camera->m_window->get_delta_time());
		//				break;
		//			}

		//		}
		//	}

		//	constexpr auto get_cross(const fan::vec2& a, const fan::vec3& b) {
		//		return fan::math::cross(fan::vec3(a.x, a.y, 0), b);
		//	}

		//	constexpr fan::vec2 get_normal(const fan::vec2& src, const fan::vec2& dst) {
		//		return get_cross(dst - src, fan::da_t<f32_t, 3>(0, 0, 1));
		//	}

		//	inline void change_sign(const fan::vec2& haluttu, fan::vec2& muutettava) {
		//		if (!fan::math::sign(haluttu.x, muutettava.x)) {
		//			muutettava.x *= -1;
		//		}
		//		if (!fan::math::sign(haluttu.y, muutettava.y)) {
		//			muutettava.y *= -1;
		//		}
		//	}


		//	inline float area(int x1, int y1, int x2, int y2, int x3, int y3)
		//	{
		//		return abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2))/2.0);
		//	}

		//	/* A function to check whether point P(x, y) lies inside the triangle formed
		//	by A(x1, y1), B(x2, y2) and C(x3, y3) */
		//	inline bool isInside(int x1, int y1, int x2, int y2, int x3, int y3, int x, int y)
		//	{  
		//		/* Calculate area of triangle ABC */
		//		float A = area (x1, y1, x2, y2, x3, y3);

		//		/* Calculate area of triangle PBC */  
		//		float A1 = area (x, y, x2, y2, x3, y3);

		//		/* Calculate area of triangle PAC */  
		//		float A2 = area (x1, y1, x, y, x3, y3);

		//		/* Calculate area of triangle PAB */   
		//		float A3 = area (x1, y1, x2, y2, x, y);

		//		/* Check if sum of A1, A2 and A3 is same as A */
		//		return (A == A1 + A2 + A3);
		//	}

		//	inline bool is_point_inside_rotated_rectangle(const fan::vec2& top_left, const fan::vec2& top_right, const fan::vec2& bottom_left, const fan::vec2& bottom_right, const fan::vec2& point) {
		//		return isInside(top_left.x, top_left.y, top_right.x, top_right.y, bottom_left.x, bottom_left.y, point.x, point.y) || isInside(bottom_right.x, bottom_right.y, bottom_left.x, bottom_left.y, top_right.x, top_right.y, point.x, point.y);
		//	}

		//	//static fan::vec2 rotate_to_no_rotate(const fan::vec2 )

		//	static void resolve_rotational_static_collision(fan_2d::opengl::rectangle& player, fan_2d::opengl::rectangle& walls, f64_t time) {	
		//		auto delta = player.get_velocity(0) * time;

		//		const auto player_points = get_velocity_corners_3d(delta);

		//		auto player_corners = player.get_corners();

		//		static constexpr std::array<std::pair<int, int>, 4> side_ids{
		//				std::pair<int, int>{ 0, 1 },
		//				std::pair<int, int>{ 1, 3 },
		//				std::pair<int, int>{ 3, 2 },
		//				std::pair<int, int>{ 2, 0 }
		//		};

		//		f64_t smallest_distance = fan::math::inf;

		//		int closest_wall = -1;

		//		int side_id = -1;

		//		for (uintptr_t i = 0; i < walls.size(); ++i) {

		//			auto wall_corners = walls.get_corners(i);

		//			for (const auto p : player_points) {

		//				for (int j = 0; j < 4; j++) { // 4 corners
		//					const fan::vec2 ray_point = fan::math::intersection_point(
		//						player_corners[p],
		//						player_corners[p] + delta,
		//						wall_corners[side_ids[j].first],
		//						wall_corners[side_ids[j].second],
		//						false
		//					);

		//					if (fan::math::ray_hit(ray_point))
		//					{

		//						auto ray_point_distance = fan_2d::math::distance(player_corners[p], ray_point);

		//						if (ray_point_distance <= smallest_distance) {
		//							smallest_distance = ray_point_distance;
		//							closest_wall = i;
		//							side_id = j;
		//						}
		//					}
		//				}
		//			}
		//		}

		//		if (closest_wall == -1) {
		//			// delta is not changed

		//			
		//		}
		//		else {

		//			// delta is changed depends about collision
		//			
		//			const auto wall_corners = walls.get_corners(closest_wall);

		//			auto a = delta;
		//			auto b = get_normal(wall_corners[side_ids[side_id].first], wall_corners[side_ids[side_id].second]);
		//			auto c = wall_corners[side_ids[side_id].second] - wall_corners[side_ids[side_id].first];
		//			
		//			auto something = sqrt((delta[0] * delta[0]) + (delta[1] * delta[1]));

		//			auto new_delta = c.normalize() * something;

		//			delta = new_delta;

		//			auto x1 = fan::math::aim_angle(wall_corners[side_ids[side_id].first], wall_corners[side_ids[side_id].second]);
		//			auto x2 = fan::math::aim_angle(player.get_position(), player.get_position() + delta);

		//			/*if ((x1 - x2)   0) {
		//				delta += b.normalize();
		//			}*/

		//		/*	for (int k = 0; k < 4; k++) {
		//				if (is_point_inside_rotated_rectangle(c1.top_left, c1.top_right, c1.bottom_left, c1.bottom_right, c2[k] + delta) {

		//				}
		//				if (is_point_inside_rotated_rectangle(c1.top_left, c1.top_right, c1.bottom_left, c1.bottom_right, c2[k] + delta) {

		//				}
		//			}*/

		//			//fan::print(delta);
		//		}

		//		player.set_position(0, player.get_position() + delta);

		//	}
		}
	}
}