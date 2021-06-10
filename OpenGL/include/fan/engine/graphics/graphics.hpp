#pragma once

#include <fan/graphics/graphics.hpp>
#include <fan/physics/physics.hpp>

namespace fan_2d {

	fan::vec2 get_body_position(fan_2d::body* body) {
		return fan::vec2(body->GetPosition()) * meter_scale;
	}

	f32_t get_body_angle(fan_2d::body* body) {
		return -body->GetAngle();
	}

	namespace engine {
		
		struct engine_t {

			fan::window window;
			fan::camera camera;
			fan_2d::world* world = nullptr;

			engine_t(const fan::vec2& gravity) : camera(&window), world(new fan_2d::world(gravity.b2())) {}

			~engine_t() {
				if (world) {
					delete world;
				}
			}

			void step(f32_t time_step) {
				world->Step(time_step, 6, 2);
			}

		};

		template <typename graphics_t, typename physics_t>
		struct base_engine : public graphics_t, public physics_t {

			base_engine(fan_2d::engine::engine_t* engine) : graphics_t(&engine->camera), physics_t(engine->world) { }

			void set_rotation(uint_t i, f_t angle, bool queue = false) {

				graphics_t::set_rotation(i, angle, queue);

				physics_t::set_rotation(i, angle);

			}

			void erase(uint_t i, bool queue = false) {
				graphics_t::erase(i);
				physics_t::erase(i);
			}
			void erase(uint_t begin, uint_t end, bool queue = false) {
				graphics_t::erase(begin, end);
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
					this->set_position(i, fan::vec2(this->get_body(i)->GetPosition()) * meter_scale);
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

				for (int i = 0; i < fan_2d::graphics::circle::size(); i++) {
					this->set_position(i, fan::vec2(this->get_body(i)->GetPosition()) * meter_scale);
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

		struct motor : public fan_2d::physics::motor {
		public:

			motor(fan_2d::engine::engine_t* engine) : fan_2d::physics::motor(engine->world) {

			}

			void push_back(fan_2d::body* a_body, fan_2d::body* b_body) {
				fan_2d::physics::motor::push_back(a_body, b_body);
			}

			void erase(uint_t i) {
				fan_2d::physics::motor::erase(i);
			}
			void erase(uint_t begin, uint_t end) {
				fan_2d::physics::motor::erase(begin * 2, end * 2);
			}

			auto size() const {
				return wheel_joints.size();
			}

		};

	}

}