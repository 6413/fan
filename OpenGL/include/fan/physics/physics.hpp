#pragma once

#include <fan/math/vector.hpp>

#include <box2d/box2d.h>

namespace fan_2d {

	typedef b2World world;

	namespace physics {

		enum class body_type {
			static_body,
			kinematic_body,
			dynamic_body
		};

		struct body_property {
			f32_t m_mass;
			f32_t m_friction;
		};

		class physics_base {

		public:

			physics_base() : m_mass(0), m_friction(0) { }

			physics_base(b2World* world) : m_world(world), m_mass(0), m_friction(0) {}

			f32_t get_mass(uint32_t i) const {
				return m_mass[i];
			}
			void set_mass(uint32_t i, f32_t mass) {
				m_mass[i] = mass;
			}

			f32_t get_friction(uint32_t i) const {
				return m_friction[i];
			}
			void set_friction(uint32_t i, f32_t friction) {
				m_friction[i] = friction;
			}

			b2Body* get_body(uint32_t i) {
				return m_body[i];
			}

			b2Fixture* get_fixture(uint32_t i) {
				return m_fixture[i];
			}

			void set_position(uint32_t i, const fan::vec2& position) {
				this->m_body[i]->SetTransform((position / meters_in_pixels).b2(), this->m_body[i]->GetAngle());
				//this->m_body[i]->SetAwake(true);
			}

			void set_rotation(uint32_t i, f64_t rotation) {
				this->m_body[i]->SetTransform(this->m_body[i]->GetPosition(), -rotation);
			}

			void set_angular_rotation(uint32_t i, f64_t w) {
				this->m_body[i]->SetAngularVelocity(w);
				this->m_body[i]->SetAwake(true);
			}

		protected:

			std::vector<f64_t> m_mass;
			std::vector<f64_t> m_friction;

			b2World* m_world;

			std::vector<b2Body*> m_body;
			std::vector<b2Fixture*> m_fixture;
		};

		struct rectangle : public physics_base {

			using physics_base::physics_base;

			void push_back(const fan::vec2& position, const fan::vec2& size, body_type body_type_, fan_2d::physics::body_property body_property = { 1, 10 }) {
				b2BodyDef body_def;
				body_def.type = (b2BodyType)body_type_;
				body_def.position.Set(position.x / meters_in_pixels + size.x / 2 / meters_in_pixels, position.y / meters_in_pixels + size.y / 2 / meters_in_pixels);
				m_body.emplace_back(m_world->CreateBody(&body_def));

				b2PolygonShape shape;
				shape.SetAsBox((size.x * 0.5) / meters_in_pixels, (size.y * 0.5) / meters_in_pixels);

				b2FixtureDef fixture_def;
				fixture_def.shape = &shape;
				fixture_def.density = body_property.m_mass;
				fixture_def.friction = body_property.m_friction;

				m_fixture.emplace_back(m_body[m_body.size() - 1]->CreateFixture(&fixture_def));
			}

			void erase(uint_t i) {
				m_world->DestroyBody(get_body(i));

				m_fixture.erase(m_fixture.begin() + i);
				m_body.erase(m_body.begin() + i);

			}
			void erase(uint_t begin, uint_t end) {

				for (int i = begin; i < end; i++) {
					m_world->DestroyBody(get_body(i));
				}

				m_fixture.erase(m_fixture.begin() + begin, m_fixture.begin() + end);
				m_body.erase(m_body.begin() + begin, m_body.begin() + end);
			}

		};

		struct circle : public physics_base {

			using physics_base::physics_base;

			void push_back(const fan::vec2& position, f32_t radius, body_type body_type_, fan_2d::physics::body_property body_property = { 1, 10 }) {
				b2BodyDef body_def;
				body_def.type = (b2BodyType)body_type_;
				body_def.position.Set(position.x / meters_in_pixels, position.y / meters_in_pixels);
				m_body.emplace_back(m_world->CreateBody(&body_def));

				b2CircleShape shape;
				shape.m_p.Set(0, 0);
				shape.m_radius = radius / meters_in_pixels;
			
				b2FixtureDef fixture_def;
				fixture_def.shape = &shape;
				fixture_def.density = body_property.m_mass;
				fixture_def.friction = body_property.m_friction;

				m_fixture.emplace_back(m_body[m_body.size() - 1]->CreateFixture(&fixture_def));
			}

		};

		

	};

}