#pragma once

#include _FAN_PATH(types/vector.h)
#include _FAN_PATH(bll.h)

#include <box2d/box2d.h)
#include <box2d/b2_rope.h)

#ifdef fan_platform_windows

	#pragma comment(lib, "lib/box2d/box2d.lib")

#endif

#include <memory>

namespace fan_2d {

	typedef b2World world;
	typedef b2FixtureDef fixture_def;

	enum shape_type {
		triangle,
		rectangle,
		circle,
		convex
	};

	struct shape : public b2Shape {

	};

	struct fixture;

	namespace physics {

		enum class body_type {
			static_body,
			kinematic_body,
			dynamic_body
		};

		struct body_property {
			f32_t mass = 0;
			f32_t friction = 0;
			f32_t restitution = 0;
		};

	}

	struct body : private b2Body {

		using inherit_t = b2Body;

		using b2Body::b2Body;

		inline fan_2d::fixture* create_fixture(const fan_2d::fixture_def* fixture_def) {
			return (fan_2d::fixture*)inherit_t::CreateFixture((const b2FixtureDef*)fixture_def);
		}

		inline void apply_force(const fan::vec2& force, const fan::vec2& point = 0) {
			if (point == 0) {
				inherit_t::ApplyForceToCenter(force.b2(), true);
			}
			else {
				inherit_t::ApplyForce(force.b2(), point.b2(), true);
			}
		}

		inline void apply_impulse(const fan::vec2& force, const fan::vec2& point = 0) {
			if (point == 0) {
				inherit_t::ApplyLinearImpulseToCenter(force.b2(), true);
			}
			else {
				inherit_t::ApplyLinearImpulse(force.b2(), point.b2(), true);
			}
		}

		inline void apply_torque(f32_t torque) {
			inherit_t::ApplyTorque(torque, true);

		}

		inline f32_t get_angular_damping() const {
			return inherit_t::GetAngularDamping();
		}
		inline void set_angular_damping(f32_t amount) {
			inherit_t::SetAngularDamping(amount);
		}

		inline f32_t get_angular_velocity() const {
			return inherit_t::GetAngularVelocity();
		}
		inline void set_angular_velocity(f32_t velocity) {
			inherit_t::SetAngularVelocity(velocity);
		}

		inline fan_2d::fixture* get_fixture_list() {
			return (fan_2d::fixture*)inherit_t::GetFixtureList();
		}

		inline f32_t get_gravity_scale() const {
			return inherit_t::GetGravityScale();
		}
		inline void set_gravity_scale(f32_t scale) {
			inherit_t::SetGravityScale(scale);
		}

		inline f32_t get_velocity_damping() const {
			return inherit_t::GetLinearDamping();
		}
		inline void set_velocity_damping(f32_t amount) {
			inherit_t::SetLinearDamping(amount);
		}

		inline fan::vec2 get_velocity() const {
			return inherit_t::GetLinearVelocity();
		}
		inline void set_velocity(const fan::vec2& velocity) {
			inherit_t::SetLinearVelocity(velocity.b2());
		}

		inline fan_2d::shape_type get_shape_type() const {
			return m_shape_type;
		}
		inline void set_shape_type(fan_2d::shape_type shape_type) {
			m_shape_type = shape_type;
		}

		inline fan_2d::physics::body_type get_body_type() const {
			return (fan_2d::physics::body_type)inherit_t::GetType();
		}

		inline f32_t get_mass() const {
			return inherit_t::GetMass();
		}

		inline bool is_bullet() const {
			return inherit_t::IsBullet();
		}

		inline bool is_awake() const {
			return inherit_t::IsAwake();
		}
		inline void set_awake(bool flag) {
			return inherit_t::SetAwake(flag);
		}

		inline bool is_enabled() const {
			return inherit_t::IsEnabled();
		}
		inline void set_enabled(bool flag) {
			inherit_t::SetEnabled(flag);
		}

		inline fan::vec2 get_position() const {
			return inherit_t::GetPosition();
		}

		inline f32_t get_angle() const {
			return inherit_t::GetAngle();
		}

		inline void set_transform(const fan::vec2& position, f32_t angle) {
			inherit_t::SetTransform(position.b2(), angle);
		}

		inline fan_2d::world* get_world() {
			return (fan_2d::world*)inherit_t::GetWorld();
		}

		void set_user_data(uintptr_t data) {
			b2BodyUserData user_data;
			user_data.pointer = data;
			inherit_t::GetUserData() = user_data;
		}

		shape_type m_shape_type;
	};

	struct aabb_t : b2AABB {

	};

	struct fixture : private b2Fixture {

		fixture() {}

		using inherit_t = b2Fixture;

		using b2Fixture::b2Fixture;

		inline fan_2d::body* get_body() {
			return (fan_2d::body*)m_body;
		}
		inline f32_t get_density() const {
			return inherit_t::GetDensity();
		}
		inline void set_density(f32_t density) {
			inherit_t::SetDensity(density);
		}

		inline f32_t get_restitution() const {
			return inherit_t::GetRestitution();
		}
		inline void set_restitution(f32_t restitution) {
			inherit_t::SetRestitution(restitution);
		}

		inline f32_t get_friction() const {
			return inherit_t::GetFriction();
		}
		inline void set_friction(f32_t friction) {
			inherit_t::SetFriction(friction);
		}

		inline fan_2d::shape_type get_shape_type() {
			return get_body()->get_shape_type();
		}
		inline void set_shape_type(fan_2d::shape_type shape_type) {
			return get_body()->set_shape_type(shape_type);
		}

		inline fan_2d::shape* get_shape() {
			return (fan_2d::shape*)inherit_t::GetShape();
		}

		inline fan_2d::physics::body_type get_body_type() {
			return (fan_2d::physics::body_type)inherit_t::GetType();
		}

		using inherit_t::RayCast;

		inline bool is_sensor() const {
			return inherit_t::IsSensor();
		}
		inline void set_sensor(bool flag) {
			inherit_t::SetSensor(flag);
		}

		inline uint32_t get_index() const {
			return m_userData.pointer;
		}

		inline void set_user_data(uintptr_t data) {
			b2FixtureUserData user_data;
			user_data.pointer = data;
			b2Fixture::GetUserData() = user_data;
		}

	};

	struct world_manifold : public b2WorldManifold {

	};

	struct manifold : public b2Manifold {

	};

	struct contact : private b2Contact {

		using inherit_t = b2Contact;

		inline fan_2d::manifold* get_manifold() {
			return (fan_2d::manifold*)inherit_t::GetManifold();
		}
		inline const fan_2d::manifold* get_manifold() const {
			return (const fan_2d::manifold*)inherit_t::GetManifold();
		}

		inline void get_world_manifold(fan_2d::world_manifold* world_manifold) const {
			inherit_t::GetWorldManifold((b2WorldManifold*)world_manifold);
		}

		inline bool is_touching() const {
			return inherit_t::IsTouching();
		}

		inline void set_enabled(bool flag) {
			inherit_t::SetEnabled(flag);
		}

		inline bool is_enabled() const {
			return inherit_t::IsEnabled();
		}

		inline fan_2d::contact* get_next() {
			return (fan_2d::contact*)inherit_t::GetNext();
		}
		inline const fan_2d::contact* get_next() const {
			return (const fan_2d::contact*)inherit_t::GetNext();
		}

		fan_2d::fixture* get_fixture_a() {
			return (fan_2d::fixture*)inherit_t::GetFixtureA();
		}
		const fan_2d::fixture* get_fixture_a() const {
			return (const fan_2d::fixture*)inherit_t::GetFixtureA();
		}

		int32 get_child_index_a() const {
			return inherit_t::GetChildIndexA();
		}

		fan_2d::fixture* get_fixture_b() {
			return (fan_2d::fixture*)inherit_t::GetFixtureB();
		}
		const fan_2d::fixture* get_fixture_b() const {
			return (const fan_2d::fixture*)inherit_t::GetFixtureB();
		}

		int32 get_child_index_b() const {
			return inherit_t::GetChildIndexB();
		}

		void set_friction(f32_t friction) {
			inherit_t::SetFriction(friction);
		}

		f32_t get_friction() const {
			return inherit_t::GetFriction();
		}

		void reset_friction() {
			inherit_t::ResetFriction();
		}

		void set_restitution(f32_t restitution) {
			inherit_t::SetRestitution(restitution);
		}

		f32_t get_restitution() const {
			return inherit_t::GetRestitution();
		}

		void reset_restitution() {
			inherit_t::ResetFriction();
		}

		void set_restitution_threshold(f32_t threshold) {
			inherit_t::SetRestitutionThreshold(threshold);
		}

		f32_t get_restitution_threshold() const {
			inherit_t::GetRestitutionThreshold();
		}

		void reset_restitution_threshold() {
			inherit_t::ResetRestitutionThreshold();
		}

		void set_tanget_speed(float speed) {
			inherit_t::SetTangentSpeed(speed * meter_scale);
		}

		f32_t get_tanget_speed() const {
			return inherit_t::GetTangentSpeed();
		}

		virtual void Evaluate(b2Manifold* manifold, const b2Transform& xfA, const b2Transform& xfB) = 0;
	};

	namespace physics {

		
		template <typename pfixture_t, typename pbody_t>
		class physics_callbacks : public b2ContactListener {

		public:

			using on_collision_t = std::function<void(fan_2d::fixture* a, fan_2d::fixture* b)>;
			using on_presolve_t = std::function<void(fan_2d::contact* contact, const fan_2d::manifold* old_manifold)>;

			void set_callback_on_collision(const on_collision_t& function) {
				m_on_collision = function;
			}

			void set_callback_on_collision_exit(const on_collision_t& function) {
				m_on_collision_exit = function;
			}

			void set_callback_on_pre_solve(const on_presolve_t& function) {
				m_on_presolve = function;
			}

			inline fan_2d::body* get_body(uint32_t i) {
				return m_body[i];
			}

			inline fan_2d::fixture* get_fixture(uint32_t i) {
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

			void PreSolve(b2Contact* contact, const b2Manifold* oldManifold) {
				if (m_on_presolve) {
					m_on_presolve((fan_2d::contact*)contact, (const fan_2d::manifold*)oldManifold);
				}
			}
			
			pbody_t m_body;
			pfixture_t m_fixture;

		private:

			on_collision_t m_on_collision;
			on_collision_t m_on_collision_exit;
			on_presolve_t m_on_presolve;

		};

		template <typename pfixture_t, typename pbody_t>
		class physics_base : public physics_callbacks<pfixture_t, pbody_t> {

		public:

			physics_base() { }

			physics_base(b2World* world) : m_world(world) {
				m_world->SetContactListener(this);
			}

			using inherit_t = physics_callbacks<pfixture_t, pbody_t>;

			// check if body should be woken

			void apply_force(uint32_t i, const fan::vec2& force, const fan::vec2& point = 0) {
				inherit_t::m_body[i]->apply_force(force * meter_scale, point);
			}

			void apply_impulse(uint32_t i, const fan::vec2& force, const fan::vec2& point = 0) {
				inherit_t::m_body[i]->apply_impulse(force * meter_scale, point);
			}

			void apply_torque(uint32_t i, f32_t torque) {
				inherit_t::m_body[i]->apply_torque(torque * meter_scale);
			}

			f32_t get_angular_damping(uint32_t i) const {
				return inherit_t::m_body[i]->get_angular_damping() / meter_scale;
			}
			void set_angular_damping(uint32_t i, f32_t amount) {
				inherit_t::m_body[i]->set_angular_damping(amount * meter_scale);
			}

			f32_t get_angular_velocity(uint32_t i) const {
				return inherit_t::m_body[i]->get_angular_velocity() / meter_scale;
			}
			void set_angular_velocity(uint32_t i, f32_t velocity) const {
				inherit_t::m_body[i]->set_angular_velocity(velocity * meter_scale);
			}

			fan_2d::fixture* get_fixture_list(uint32_t i) {
				return (fan_2d::fixture*)inherit_t::m_body[i]->get_fixture_list();
			}

			f32_t get_gravity_scale(uint32_t i) const {
				return inherit_t::m_body[i]->get_gravity_scale() / meter_scale;
			}
			void set_gravity_scale(uint32_t i, f32_t scale) {
				inherit_t::m_body[i]->set_gravity_scale(scale * meter_scale);
			}

			f32_t get_velocity_damping(uint32_t i) const {
				return inherit_t::m_body[i]->get_velocity_damping() / meter_scale;
			}
			void set_velocity_damping(uint32_t i, f32_t amount) {
				inherit_t::m_body[i]->set_velocity_damping(amount * meter_scale);
			}

			fan::vec2 get_velocity(uint32_t i) const {
				return inherit_t::m_body[i]->get_velocity() / meter_scale;
			}
			void set_velocity(uint32_t i, const fan::vec2& velocity) {
				inherit_t::m_body[i]->set_velocity(velocity * meter_scale);
			}

			body_type get_body_type(uint32_t i) const {
				return (body_type)inherit_t::m_body[i]->get_body_type();
			}

			f32_t get_density(uint32_t i) {
				return inherit_t::m_fixture[i]->get_density();
			}
			void set_density(uint32_t i, f32_t density) {
				inherit_t::m_fixture[i]->set_density(density);
			}

			f32_t get_mass(uint32_t i) {
				return inherit_t::m_body[i]->get_mass();
			}

			bool is_bullet(uint32_t i) const {
				return inherit_t::m_body[i]->is_bullet();
			}

			bool is_awake(uint32_t i) const {
				return inherit_t::m_body[i]->is_awake();
			}
			void set_awake(uint32_t i, bool flag) {
				return inherit_t::m_body[i]->set_awake(flag);
			}

			bool is_enabled(uint32_t i) const {
				return inherit_t::m_body[i]->is_enabled();
			}
			void set_enabled(uint32_t i, bool flag) {
				inherit_t::m_body[i]->set_enabled(flag);
			}

			f32_t get_restitution(uint32_t i) const {
				return inherit_t::m_fixture[i]->get_restitution() / meter_scale;
			}
			void set_restitution(uint32_t i, f32_t restitution) {
				inherit_t::m_fixture[i]->set_restitution(restitution * meter_scale);
			}

			void set_position(uint32_t i, const fan::vec2& position) {
				this->inherit_t::m_body[i]->set_transform((position / meter_scale).b2(), this->inherit_t::m_body[i]->get_angle());
				this->inherit_t::m_body[i]->set_awake(true);
			}

			void set_angle(uint32_t i, f64_t rotation) {
				this->inherit_t::m_body[i]->set_transform(this->inherit_t::m_body[i]->get_position(), -rotation);
			}

			void set_angular_rotation(uint32_t i, f64_t w) {
				this->inherit_t::m_body[i]->set_angular_velocity(w);
				this->inherit_t::m_body[i]->set_awake(true);
			}

			void erase(uintptr_t i) {
				

				m_world->DestroyBody((b2Body*)inherit_t::get_body(i));

				if constexpr(std::is_same_v<pfixture_t, std::vector<fan_2d::fixture*>>) {
					inherit_t::m_fixture.erase(inherit_t::m_fixture.begin() + i);
					inherit_t::m_body.erase(inherit_t::m_body.begin() + i);
				}
				else {
					inherit_t::m_fixture.unlink(i);
					inherit_t::m_body.unlink(i);
				}


			}
			void erase(uintptr_t begin, uintptr_t end) {

				if (begin > inherit_t::m_body.size() || end > inherit_t::m_body.size()) {
					return;
				}

				for (int i = begin; i < end; i++) {
					m_world->DestroyBody((b2Body*)inherit_t::get_body(i));
				}

				inherit_t::m_fixture.erase(inherit_t::m_fixture.begin() + begin, inherit_t::m_fixture.begin() + end);
				inherit_t::m_body.erase(inherit_t::m_body.begin() + begin, inherit_t::m_body.begin() + end);
			}

		protected:

			b2World* m_world;

		};

		struct triangle_t : physics_base<std::vector<fan_2d::fixture*>, std::vector<fan_2d::body*>> {

			using physics_base::physics_base;

			struct properties_t {
				fan::vec2 position;

				fan::vec2 points[3];

				f32_t angle;

				fan_2d::physics::body_type body_type;
				fan_2d::physics::body_property body_property = { 10, 1, 0.1 };
			};

			void push_back(const properties_t& properties) {
				b2BodyDef body_def;
				body_def.type = (b2BodyType)properties.body_type;
				body_def.position.Set(properties.position.x / meter_scale, properties.position.y / meter_scale);
				body_def.angle = -properties.angle;
				m_body.emplace_back((fan_2d::body*)m_world->CreateBody(&body_def));

				b2PolygonShape shape;
				shape.Set((const b2Vec2*)properties.points, 3);

				b2FixtureDef fixture_def;
				fixture_def.shape = &shape;
				fixture_def.density = properties.body_property.mass;
				fixture_def.friction = properties.body_property.friction;
				fixture_def.restitution = properties.body_property.restitution;

				m_fixture.emplace_back((fan_2d::fixture*)m_body[m_body.size() - 1]->create_fixture(&fixture_def));

				m_fixture[m_fixture.size() - 1]->set_user_data(m_fixture.size() - 1);
				m_fixture[m_fixture.size() - 1]->set_shape_type(fan_2d::shape_type::triangle);
			}

		};

		struct rectangle_t : physics_base<std::vector<fan_2d::fixture*>, std::vector<fan_2d::body*>> {

			using physics_base::physics_base;

			struct properties_t {
				fan::vec2 position;
				fan::vec2 size;

				f32_t angle;

				fan_2d::physics::body_type body_type;
				fan_2d::physics::body_property body_property = { 10, 1, 0.1 };
			};

			void push_back(const properties_t& properties) {
				b2BodyDef body_def;
				body_def.type = (b2BodyType)properties.body_type;
				body_def.position.Set((properties.position.x + properties.size.x * 0.5) / meter_scale, (properties.position.y + properties.size.y * 0.5) / meter_scale);
				body_def.angle = -properties.angle;
				m_body.emplace_back((fan_2d::body*)m_world->CreateBody(&body_def));

				b2PolygonShape shape;
				shape.SetAsBox((properties.size.x * 0.5) / meter_scale, (properties.size.y * 0.5) / meter_scale);

				b2FixtureDef fixture_def;
				fixture_def.shape = &shape;
				fixture_def.density = properties.body_property.mass;
				fixture_def.friction = properties.body_property.friction;
				fixture_def.restitution = properties.body_property.restitution;

				m_fixture.emplace_back((fan_2d::fixture*)m_body[m_body.size() - 1]->create_fixture(&fixture_def));

				m_fixture[m_fixture.size() - 1]->set_user_data(m_fixture.size() - 1);
				m_fixture[m_fixture.size() - 1]->set_shape_type(fan_2d::shape_type::rectangle);
			}

		};

		struct circle_t : public physics_base<std::vector<fan_2d::fixture*>, std::vector<fan_2d::body*>> {

			using physics_base::physics_base;

			struct properties_t {
				fan::vec2 position;
				f32_t radius;
				body_type body_type;
				body_property body_property = { 1, 10, 0.1 };
			};

			void push_back(const properties_t& properties) {
				b2BodyDef body_def;
				body_def.type = (b2BodyType)properties.body_type;
				body_def.position.Set(properties.position.x / meter_scale, properties.position.y / meter_scale);
				m_body.emplace_back((fan_2d::body*)m_world->CreateBody(&body_def));

				b2CircleShape shape;
				shape.m_p.Set(0, 0);
				shape.m_radius = properties.radius / meter_scale;
			
				b2FixtureDef fixture_def;
				fixture_def.shape = &shape;
				fixture_def.density = properties.body_property.mass;
				fixture_def.friction = properties.body_property.friction;

				m_fixture.emplace_back((fan_2d::fixture*)m_body[m_body.size() - 1]->create_fixture(&fixture_def));
				m_fixture[m_fixture.size() - 1]->set_user_data(m_fixture.size() - 1);
				m_fixture[m_fixture.size() - 1]->set_shape_type(fan_2d::shape_type::circle);
			}

			void set_angle(uint32_t i, f32_t angle) {

			}
		};
		
		struct sprite : public physics_base<bll_t<fan_2d::fixture*>, bll_t<fan_2d::body*>> {

			using physics_base::physics_base;

			struct properties_t {
				fan::vec2 position;
				f32_t radius;
				body_type body_type;
				body_property body_property = { 1, 10, 0.1 };
			};

		protected:
			//fan_2d::physics::triangle* physics_rectangle;

		public:

			uint32_t physics_size() const {
				return m_fixture.size();
			}

			uint32_t physics_add_triangle(uint32_t physics_id, uint32_t sprite_id, const fan_2d::physics::triangle::properties_t& properties) {

				if (!m_body[physics_id]) {

					b2BodyDef body_def;
					body_def.type = (b2BodyType)properties.body_type;
					body_def.position.Set(properties.position.x / meter_scale, properties.position.y / meter_scale);
					body_def.angle = -properties.angle;

					m_body[physics_id] = (fan_2d::body*)m_world->CreateBody(&body_def);
				}
				
				b2PolygonShape shape;

				fan::vec2* converted_points = (fan::vec2*)properties.points;

				for (int i = 0; i < 3; i++) {
					converted_points[i] = properties.points[i] / meter_scale;
				}

				// maybe need to set center offset with m_centroid if we want multiple fixtures
				shape.Set((const b2Vec2*)converted_points, 3);

				b2FixtureDef fixture_def;
				fixture_def.shape = &shape;
				fixture_def.density = properties.body_property.mass;
				fixture_def.friction = properties.body_property.friction;
				fixture_def.restitution = properties.body_property.restitution;
				auto id = m_fixture.new_node_last();
				m_fixture[id] = (fan_2d::fixture*)m_body[physics_id]->create_fixture(&fixture_def);

				m_fixture[id]->set_user_data(sprite_id);
				m_fixture[id]->set_shape_type(fan_2d::shape_type::triangle);

				return id;
			}
			
			uint32_t physics_add_rectangle(const fan::vec2& point, uint32_t physics_reference, uint32_t sprite_id, const fan_2d::physics::rectangle::properties_t& properties) {

				b2PolygonShape shape;

				// we want to initialize only once and if this if is not true it means we have multiple fixtures
				if (!m_body[physics_reference]) {

					b2BodyDef body_def;
					body_def.type = (b2BodyType)properties.body_type;
					body_def.position.Set((point.x / meter_scale), (point.y / meter_scale));
					body_def.angle = -properties.angle;

					m_body[physics_reference] = ((fan_2d::body*)m_world->CreateBody(&body_def));

					shape.SetAsBox((properties.size.x * 0.5) / meter_scale, (properties.size.y * 0.5) / meter_scale, (((properties.position + properties.size / 2) - point) / meter_scale).b2(), -properties.angle);
				}
				else {

					shape.SetAsBox((properties.size.x * 0.5) / meter_scale, (properties.size.y * 0.5) / meter_scale, (((properties.position + properties.size / 2) - point) / meter_scale).b2(), -properties.angle);
				}

				
				
				b2FixtureDef fixture_def;
				fixture_def.shape = &shape;
				fixture_def.density = properties.body_property.mass;
				fixture_def.friction = properties.body_property.friction;
				fixture_def.restitution = properties.body_property.restitution;

				auto id = m_fixture.new_node_last();

				m_fixture[id] = ((fan_2d::fixture*)m_body[physics_reference]->create_fixture(&fixture_def));

				m_fixture[id]->set_user_data(sprite_id);
				m_fixture[id]->set_shape_type(fan_2d::shape_type::rectangle);

				return id;
			}
			/*
			void physics_add_circle(uint32_t i, const fan_2d::physics::circle::properties_t& properties) {
				if (i >= m_body_amount_in_fixture.size()) {
					m_body_amount_in_fixture.resize(i + 1);
				}

				m_body_amount_in_fixture[i]++;

				b2BodyDef body_def;
				body_def.type = (b2BodyType)properties.body_type;
				body_def.position.Set(properties.position.x / meter_scale, properties.position.y / meter_scale);
				m_body.emplace_back((fan_2d::body*)m_world->CreateBody(&body_def));

				b2CircleShape shape;
				shape.m_p.Set(0, 0);
				shape.m_radius = properties.radius / meter_scale;

				b2FixtureDef fixture_def;
				fixture_def.shape = &shape;
				fixture_def.density = properties.body_property.mass;
				fixture_def.friction = properties.body_property.friction;

				if (i >= m_fixture.size()) {
					m_fixture.resize(i + 1);
				}

				m_fixture[i] = ((fan_2d::fixture*)m_body[m_body.size() - 1]->create_fixture(&fixture_def));

				m_fixture[i]->set_user_data(m_fixture.size() - 1);
				m_fixture[i]->set_shape_type(fan_2d::shape_type::circle);
			}*/


		};

		struct convex : public physics_base<std::vector<fan_2d::fixture*>, std::vector<fan_2d::body*>> {

			using physics_base::physics_base;

			struct properties_t {

				fan::vec2 position;

				fan::vec2* points;
				// maximum of 8 points per push
				uint8_t points_amount;

				f32_t angle;

				body_type body_type;
				fan_2d::physics::body_property body_property = { 1, 10, 0.1 };
			};

			void push_back(const properties_t& properties) {

				b2BodyDef body_def;
				body_def.type = b2BodyType::b2_dynamicBody;
				body_def.position.Set(properties.position.x / meter_scale, properties.position.y / meter_scale);
				body_def.angle = -properties.angle;
				m_body.emplace_back((fan_2d::body*)m_world->CreateBody(&body_def));

				for (int i = 0; i < properties.points_amount; i++) {
					properties.points[i] = properties.points[i] / meter_scale;
				}

				b2PolygonShape shape;
				shape.Set((const b2Vec2*)properties.points, properties.points_amount);

				b2FixtureDef fixture_def;
				fixture_def.shape = &shape;
				fixture_def.density = properties.body_property.mass;
				fixture_def.friction = properties.body_property.friction;
				fixture_def.restitution = properties.body_property.restitution;

				m_fixture.emplace_back((fan_2d::fixture*)m_body[m_body.size() - 1]->create_fixture(&fixture_def));

				m_fixture[m_fixture.size() - 1]->set_user_data(m_fixture.size() - 1);
				m_fixture[m_fixture.size() - 1]->set_shape_type(fan_2d::shape_type::convex);
			}
		};

		struct motor_joint : public physics_base<std::vector<fan_2d::fixture*>, std::vector<fan_2d::body*>> {

			motor_joint(fan_2d::world* world) : physics_base(world) {}

			void push_back(fan_2d::body* body_a, fan_2d::body* body_b) {

				b2WheelJointDef jd;
 
				jd.Initialize((b2Body*)body_a, (b2Body*)body_b, body_b->get_position().b2(), b2Vec2(0.0f, 1.0f));

				jd.motorSpeed = 0;
				jd.maxMotorTorque = 10000;
				jd.enableMotor = true;
				jd.lowerTranslation = -3.0f;
				jd.upperTranslation = 3.0f;
				jd.enableLimit = true;
				jd.stiffness = 1;
				jd.damping = 1;


				float hertz = 10;
				float dampingRatio = 0.7;

				b2LinearStiffness(jd.stiffness, jd.damping, hertz, dampingRatio, (b2Body*)body_a, (b2Body*)body_b);

				wheel_joints.emplace_back((b2WheelJoint*)m_world->CreateJoint(&jd));
			}

			void erase(uintptr_t i) {
				physics_base::erase(i);
				m_world->DestroyJoint(wheel_joints[i]);
				wheel_joints.erase(wheel_joints.begin() + i);
			}
			void erase(uintptr_t begin, uintptr_t end) {

				physics_base::erase(begin, end);

				for (int i = begin; i < end; i++) {
					m_world->DestroyJoint(wheel_joints[i]);
				}

				wheel_joints.erase(wheel_joints.begin() + begin, wheel_joints.begin() + end);
			}

			auto get_body_a(uint32_t i) {
				return wheel_joints[i]->GetBodyA();
			}

			auto get_body_b(uint32_t i) {
				return wheel_joints[i]->GetBodyB();
			}

			void set_speed(uint32_t i, f32_t speed) {
				wheel_joints[i]->SetMotorSpeed(speed);
			}

			auto size() const {
				return wheel_joints.size();
			}

		protected:

			using physics_base::get_body;
			using physics_base::get_fixture;

			std::vector<b2WheelJoint*> wheel_joints;

		};

	};

}