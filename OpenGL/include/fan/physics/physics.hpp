#pragma once

#include <fan/types/vector.hpp>

#include <box2d/box2d.h>
#include <box2d/b2_rope.h>

#ifdef FAN_PLATFORM_WINDOWS

	#pragma comment(lib, "box2d.lib")

#endif

#include <memory>

namespace fan_2d {

	typedef b2World world;
	typedef b2Body body;

	struct fixture : public b2Fixture {

		fixture() {}

		using b2Fixture::b2Fixture;

		uint32_t get_ptr_index() const {
			return m_userData.pointer;
		}

	};

	namespace physics {

		enum class body_type {
			static_body,
			kinematic_body,
			dynamic_body
		};

		struct body_property {
			f32_t m_mass;
			f32_t m_friction;
			f32_t m_restitution;
		};

		class physics_callbacks : public b2ContactListener {

			using on_collision_t = std::function<void(fan_2d::fixture* a, fan_2d::fixture* b)>;

		public:

			void set_callback_on_collision(const on_collision_t& function) {
				m_on_collision = function;
			}

			void set_callback_on_collision_exit(const on_collision_t& function) {
				m_on_collision_exit = function;
			}

			fan_2d::body* get_body(uint32_t i) {
				return m_body[i];
			}

			fan_2d::fixture* get_fixture(uint32_t i) {
				return m_fixture[i];
			}

		protected:

			void BeginContact(b2Contact* contact) {
				if (m_on_collision) {
					m_on_collision((fan_2d::fixture*)contact->GetFixtureA(), (fan_2d::fixture*)contact->GetFixtureB());
				}	
			}

			void EndContact(b2Contact* contact) {
				if (m_on_collision_exit) {
					m_on_collision_exit((fan_2d::fixture*)contact->GetFixtureA(), (fan_2d::fixture*)contact->GetFixtureB());
				}
			}
			
			std::vector<fan_2d::body*> m_body;
			std::vector<fan_2d::fixture*> m_fixture;

		private:

			on_collision_t m_on_collision;
			on_collision_t m_on_collision_exit;

		};

		class physics_base : public physics_callbacks {

		public:

			physics_base() : m_mass(0), m_friction(0) { }

			physics_base(b2World* world) : m_world(world), m_mass(0), m_friction(0) {
				m_world->SetContactListener(this);
			}

			void apply_force(uint32_t i, const fan::vec2& force) {
				m_body[i]->ApplyForceToCenter(force.b2(), false);
			}

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

			f32_t get_restitution(uint32_t i) const {
				return m_fixture[i]->GetRestitution();
			}
			void set_restitution(uint32_t i, f32_t restitution) {
				m_fixture[i]->SetRestitution(restitution);
			}

			void set_position(uint32_t i, const fan::vec2& position) {
				this->m_body[i]->SetTransform((position / meters_in_pixels).b2(), this->m_body[i]->GetAngle());
				this->m_body[i]->SetAwake(true);
			}

			void set_rotation(uint32_t i, f64_t rotation) {
				this->m_body[i]->SetTransform(this->m_body[i]->GetPosition(), -rotation);
			}

			void set_angular_rotation(uint32_t i, f64_t w) {
				this->m_body[i]->SetAngularVelocity(w);
				this->m_body[i]->SetAwake(true);
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

		protected:

			std::vector<f64_t> m_mass;
			std::vector<f64_t> m_friction;

			b2World* m_world;

		};

		struct rectangle : public physics_base {

			using physics_base::physics_base;

			void push_back(const fan::vec2& position, const fan::vec2& size, body_type body_type_, fan_2d::physics::body_property body_property = { 1, 10, 0.1 }) {
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
				fixture_def.restitution = body_property.m_restitution;

				m_fixture.emplace_back((fan_2d::fixture*)m_body[m_body.size() - 1]->CreateFixture(&fixture_def));

				b2FixtureUserData data;
				data.pointer = m_fixture.size() - 1;
				m_fixture[m_fixture.size() - 1]->GetUserData() = data;
			}

		};

		struct circle : public physics_base {

			using physics_base::physics_base;

			void push_back(const fan::vec2& position, f32_t radius, body_type body_type_, fan_2d::physics::body_property body_property = { 1, 10, 0.1 }) {
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

				m_fixture.emplace_back((fan_2d::fixture*)m_body[m_body.size() - 1]->CreateFixture(&fixture_def));
			}

		};

		struct rope : public physics_base, protected rectangle {

			rope(b2World* world) : physics_base(world), rectangle(world) { }

			void push_back(std::vector<fan::vec2>& joints) {

				b2RopeDef def;
				def.vertices = (b2Vec2*)joints.data();
				def.count = joints.size();
				def.gravity.Set(rectangle::m_world->GetGravity().x, rectangle::m_world->GetGravity().y);
				std::vector<f32_t> masses(joints.size());
				std::fill(masses.begin(), masses.end(), 1);
				def.masses = masses.data();

				b2RopeTuning tuning;

				tuning.bendHertz = 30.0f;
				tuning.bendDamping = 4.0f;
				tuning.bendStiffness = 1.0f;
				tuning.bendingModel = b2_pbdTriangleBendingModel;
				tuning.isometric = true;

				tuning.stretchHertz = 30.0f;
				tuning.stretchDamping = 4.0f;
				tuning.stretchStiffness = 1.0f;
				tuning.stretchingModel = b2_pbdStretchingModel;

				def.position = joints[0].b2();
				def.tuning = tuning;

				m_rope.Create(def);

				vertices = joints.data();


			//	b2Rope

				//b2Draw
				
				//rope.Step()

			}

			b2Rope m_rope;

			fan::vec2* vertices;
		};

	};

}