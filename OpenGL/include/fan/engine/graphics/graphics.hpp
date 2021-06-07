#pragma once

#include <fan/graphics/graphics.hpp>
#include <fan/physics/physics.hpp>

namespace fan_2d {

	namespace engine {
		
		struct engine_t {

			fan::window m_window;
			fan::camera m_camera;
			fan_2d::world* m_world = nullptr;

			engine_t(const fan::vec2& gravity) : m_camera(&m_window), m_world(new fan_2d::world(gravity.b2())) {}

			~engine_t() {
				if (m_world) {
					delete m_world;
				}
			}

			void step(f32_t time_step) {
				m_world->Step(time_step, 6, 2);
			}

		};

		template <typename graphics_t, typename physics_t>
		struct base_engine : public graphics_t, public physics_t {

			base_engine(fan_2d::engine::engine_t* engine) : graphics_t(&engine->m_camera), physics_t(engine->m_world) { }

			void set_rotation(uint_t i, f_t angle, bool queue = false) {

				graphics_t::set_rotation(i, angle, queue);

				physics_t::set_rotation(i, angle);

			}

			void erase(uint_t i, bool queue = false) {
				graphics_t::erase(i, queue);
				physics_t::erase(i);
			}
			void erase(uint_t begin, uint_t end, bool queue = false) {
				graphics_t::erase(begin, end, queue);
				physics_t::erase(begin, end);
			}

		};

		struct rectangle : public base_engine<fan_2d::graphics::rectangle, fan_2d::physics::rectangle> {

			using base_engine::base_engine;

			void push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, fan_2d::physics::body_type body_type, fan_2d::physics::body_property body_property = { 10, 1, 0.1 }, bool queue = false) {

				fan_2d::graphics::rectangle::push_back(position, size, color, queue);

				fan_2d::physics::rectangle::push_back(position, size, body_type, body_property);

			}

			void set_position(uint_t i, const fan::vec2& position, bool queue = false) {
				fan_2d::graphics::rectangle::set_position(i, position - this->get_size(i) / 2, queue);

				fan_2d::physics::rectangle::set_position(i, position);

			}

			void update_position() {

				for (int i = 0; i < this->size(); i++) {
					this->set_position(i, fan::vec2(this->get_body(i)->GetPosition()) * meters_in_pixels);
					this->set_rotation(i, -this->get_body(i)->GetAngle());
				}

			}

		};

		struct circle : public base_engine<fan_2d::graphics::circle, fan_2d::physics::circle> {

			using base_engine::base_engine;

			void push_back(const fan::vec2& position, f32_t radius, const fan::color& color, fan_2d::physics::body_type body_type, fan_2d::physics::body_property body_property = { 10, 0.1, 0.1 }, bool queue = false) {

				fan_2d::graphics::circle::push_back(position, radius, color, queue);

				fan_2d::physics::circle::push_back(position, radius, body_type, body_property);

			}

			void set_position(uint32_t i, const fan::vec2& position, bool queue = false) {
				fan_2d::graphics::circle::set_position(i, position, queue);

				fan_2d::physics::circle::set_position(i, position);
			}

			void update_position() {

				for (int i = 0; i < this->size(); i++) {
					this->set_position(i, fan::vec2(this->get_body(i)->GetPosition()) * meters_in_pixels);
				}

			}

		};

		struct rope : public base_engine<fan_2d::graphics::line, fan_2d::physics::rope>{

			rope(engine_t* engine, const fan::vec2& position, fan_2d::physics::body_type body_type_) : base_engine(engine) {

			}

			void push_back(std::vector<fan::vec2>& joints, const fan::color& color) {

				fan_2d::graphics::line::push_back(joints[0], joints[1], color);

				for (int i = 1; i < int(joints.size() / 2); i++) {
					fan_2d::graphics::line::push_back(joints[i], joints[i + 1], color);
				}

				fan_2d::physics::rope::push_back(joints);
			}

			fan::vec2 step() {
				b2Vec2 p;
				m_rope.Step(1.0 / 144, 6, p);

				return p;
			}

			void update_position() {

				//for (int i = 0; i < fan_2d::graphics::line::size(); i++) {
				//	fan_2d::graphics::line::set_line(
				//		i, 
				//		fan::vec2(fan_2d::physics::rope::rectangle::get_body(i)->GetPosition()) * meters_in_pixels,
				//		fan::vec2(fan_2d::physics::rope::rectangle::get_body(i + 1)->GetPosition()) * meters_in_pixels
				//	);
				//}

			}

		};

	}

}