#pragma once

#include <fan/types/vector.h>
#include <fan/bll.h>

#include <box2d/box2d.h>
#include <box2d/b2_rope.h>

#ifdef fan_platform_windows

#pragma comment(lib, "lib/box2d/box2d.lib")

#endif

#include <memory>

constexpr auto meter_scale = 100;

namespace fan_2d {

	using body_def = b2BodyDef;

	using joint_def = b2JointDef;
	using joint = b2Joint;

	using wheel_joint = b2WheelJoint;
	using contact_listener_cb = b2ContactListener;

	struct body;

	struct world : protected b2World {

		using b2World::b2World;

		using inherit_t = b2World;

		inline fan::vec2 get_gravity() const {
			return inherit_t::GetGravity();
		}

		inline void set_gravity(const fan::vec2& gravity) {
			inherit_t::SetGravity(gravity.b2());
		}

		inline void step(f32_t time_step, uint32_t velocity_iterations, uint32_t position_iterations) {
			inherit_t::Step(time_step, velocity_iterations, position_iterations);
		}
		inline bool is_locked() const {
			return inherit_t::IsLocked();
		}
		inline fan_2d::body* create_body(const body_def* body_def_) {
			return (fan_2d::body*)inherit_t::CreateBody((const b2BodyDef*)body_def_);
		}
		inline void destroy_body(fan_2d::body* body_) {
			inherit_t::DestroyBody((b2Body*)body_);
		}
		inline fan_2d::joint* create_joint(const fan_2d::joint_def* joint_def_) {
			(fan_2d::joint*)inherit_t::CreateJoint((const b2JointDef*)joint_def_);
		}
		inline void destroy_joint(fan_2d::joint* joint_) {
			inherit_t::DestroyJoint((b2Joint*)joint_);
		}
		inline void set_contact_listener(contact_listener_cb* cb) {
			inherit_t::SetContactListener((b2ContactListener*)cb);
		}

	};

	typedef b2FixtureDef fixture_def;

	enum shape_type {
		triangle,
		rectangle,
		circle,
		convex,
		edge
	};

	struct shape : public b2Shape {

	};

	struct fixture;

	namespace physics {

		enum class body_type_t {
			static_body,
			kinematic_body,
			dynamic_body
		};

		struct body_property_t {
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
				inherit_t::ApplyForceToCenter((force / meter_scale).b2(), true);
			}
			else {
				inherit_t::ApplyForce((force / meter_scale).b2(), (point / meter_scale).b2(), true);
			}
		}

		inline void apply_impulse(const fan::vec2& force, const fan::vec2& point = 0) {
			if (point == 0) {
				inherit_t::ApplyLinearImpulseToCenter((force / meter_scale).b2(), true);
			}
			else {
				inherit_t::ApplyLinearImpulse((force / meter_scale).b2(), point.b2(), true);
			}
		}

		inline void apply_torque(f32_t torque) {
			inherit_t::ApplyTorque(torque / meter_scale, true);

		}

		inline f32_t get_angular_damping() const {
			return inherit_t::GetAngularDamping() * meter_scale;
		}
		inline void set_angular_damping(f32_t amount) {
			inherit_t::SetAngularDamping(amount / meter_scale);
		}

		inline f32_t get_angular_velocity() const {
			return inherit_t::GetAngularVelocity() * meter_scale;
		}
		inline void set_angular_velocity(f32_t velocity) {
			inherit_t::SetAngularVelocity(velocity / meter_scale);
		}

		inline bool is_fixed_rotation() const {
			return inherit_t::IsFixedRotation();
		}
		inline void set_fixed_rotation(bool flag) {
			inherit_t::SetFixedRotation(flag);
		}

		inline fan_2d::fixture* get_fixture_list() {
			return (fan_2d::fixture*)inherit_t::GetFixtureList();
		}

		inline fan_2d::fixture* get_next() {
			return (fan_2d::fixture*)inherit_t::GetNext();
		}

		inline f32_t get_gravity_scale() const {
			return inherit_t::GetGravityScale() * meter_scale;
		}
		inline void set_gravity_scale(f32_t scale) {
			inherit_t::SetGravityScale(scale / meter_scale);
		}

		inline f32_t get_velocity_damping() const {
			return inherit_t::GetLinearDamping() * meter_scale;
		}
		inline void set_velocity_damping(f32_t amount) {
			inherit_t::SetLinearDamping(amount / meter_scale);
		}

		inline fan::vec2 get_velocity() const {
			return fan::vec2(inherit_t::GetLinearVelocity()) * meter_scale;
		}
		inline void set_velocity(const fan::vec2& velocity) {
			inherit_t::SetLinearVelocity((velocity / meter_scale).b2());
		}

		inline fan_2d::shape_type get_shape_type() const {
			return m_shape_type;
		}
		inline void set_shape_type(fan_2d::shape_type shape_type) {
			m_shape_type = shape_type;
		}

		inline fan_2d::physics::body_type_t get_body_type() const {
			return (fan_2d::physics::body_type_t)inherit_t::GetType();
		}

		inline f32_t get_mass() const {
			return inherit_t::GetMass() * meter_scale;
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
			return fan::vec2(inherit_t::GetPosition()) * meter_scale;
		}

		inline fan::vec2 get_world_center() const {
			return fan::vec2(inherit_t::GetWorldCenter()) * meter_scale;
		}

		inline f32_t get_angle() const {
			return inherit_t::GetAngle() * meter_scale;
		}

		inline void set_transform(const fan::vec2& position, f32_t angle) {
			inherit_t::SetTransform((position / meter_scale).b2(), angle);
		}

		inline fan_2d::world* get_world() {
			return (fan_2d::world*)inherit_t::GetWorld();
		}

		uintptr_t get_user_data() {
			return inherit_t::GetUserData().pointer;
		}

		void set_user_data(uintptr_t data) {
			b2BodyUserData user_data;
			user_data.pointer = data;
			inherit_t::GetUserData() = user_data;
		}

		inline void destroy_fixture(fan_2d::fixture* fixture) {
			inherit_t::DestroyFixture((b2Fixture*)fixture);
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
			return inherit_t::GetDensity() * meter_scale;
		}
		inline void set_density(f32_t density) {
			inherit_t::SetDensity(density / meter_scale);
		}

		inline f32_t get_restitution() const {
			return inherit_t::GetRestitution() * meter_scale;
		}
		inline void set_restitution(f32_t restitution) {
			inherit_t::SetRestitution(restitution / meter_scale);
		}

		inline f32_t get_friction() const {
			return inherit_t::GetFriction() * meter_scale;
		}
		inline void set_friction(f32_t friction) {
			inherit_t::SetFriction(friction / meter_scale);
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

		inline fan_2d::physics::body_type_t get_body_type() {
			return (fan_2d::physics::body_type_t)inherit_t::GetType();
		}

		inline fan_2d::fixture* get_next() {
			return (fan_2d::fixture*)inherit_t::GetNext();
		}

		using inherit_t::RayCast;

		inline bool is_sensor() const {
			return inherit_t::IsSensor();
		}
		inline void set_sensor(bool flag) {
			inherit_t::SetSensor(flag);
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
			return inherit_t::GetRestitutionThreshold();
		}

		void reset_restitution_threshold() {
			inherit_t::ResetRestitutionThreshold();
		}

		void set_tanget_speed(float speed) {
			inherit_t::SetTangentSpeed(speed);
		}

		f32_t get_tanget_speed() const {
			return inherit_t::GetTangentSpeed();
		}

		virtual void Evaluate(b2Manifold* manifold, const b2Transform& xfA, const b2Transform& xfB) = 0;
	};

	namespace engine {

		struct engine_t {

			fan::window_t window;
			fan::opengl::context_t context;
			fan_2d::world world;

			//std::unordered_map<void*, bool> m_to_avoid;

			std::unordered_map<void*, std::function<void()>> m_to_update;

			// functions that will be executed outside step - queue

			std::vector<std::function<void()>> m_queue_after_step;

			engine_t(const fan::vec2& gravity) : world(fan_2d::world(fan::vec2().b2())) {
				window.open();

				context.init();
				context.bind_to_window(&window);
				context.set_viewport(0, window.get_size());

				window.add_resize_callback(&context, [](fan::window_t*, const fan::vec2i& s, void* user_ptr) {
					((fan::opengl::context_t*)user_ptr)->set_viewport(0, s);
				});

			}

			void step(f32_t time_step) {
				
				world.step(time_step, 6, 2);

				for (int i = 0; i < m_queue_after_step.size(); i++) {
					m_queue_after_step[i]();
				}

				m_queue_after_step.clear();

				for (auto i : m_to_update) {
					i.second();
				}

				//m_to_avoid.clear();
			}

			//void avoid_updating(void* shape) {
			//	m_to_avoid.insert(std::make_pair(shape, false));
			//}

			void push_update(void* shape, std::function<void()> update_function) {
				m_to_update.insert(std::make_pair(shape, update_function));
			}

		};
	}

	namespace physics {

		
		struct triangle_t {

			struct properties_t {
				fan::vec2 position;

				// offsets from position
				fan::vec2 points[3];

				f32_t angle;

				fan_2d::physics::body_type_t body_type;
				fan_2d::physics::body_property_t body_property = { 10, 1, 0.1 };
			};

		};

		struct rectangle_t {

			struct properties_t {
				fan::vec2 position;
				fan::vec2 size;

				f32_t angle;

				fan_2d::physics::body_type_t body_type;
				fan_2d::physics::body_property_t body_property = { 10, 1, 0.1 };
			};
		};

		struct circle_t {

			struct properties_t {
				fan::vec2 position;
				f32_t radius;
				body_type_t body_type;
				body_property_t body_property = { 1, 10, 0.1 };
			};

		};

		struct convex_t {

			struct properties_t {

				fan::vec2 position;

				fan::vec2* points;
				// maximum of 8 points per push
				uint8_t points_amount;

				f32_t angle;

				body_type_t body_type;
				fan_2d::physics::body_property_t body_property = { 1, 10, 0.1 };
			};

		};

		struct edge_shape_t {
			
			struct properties_t {

				fan::vec2 src;
				fan::vec2 dst;

				f32_t angle = 0;

				body_type_t body_type;
				fan_2d::physics::body_property_t body_property = { 1, 10, 0.1 };
			};
		};

		template <typename user_struct>
		struct collider_callbacks_t : public b2ContactListener {

			struct node_t {
				fan_2d::body* body;
				bool removed = false;
				user_struct user_data;
			};

			bll_t<node_t, uint32_t> collider_list;

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

			inline fan_2d::body* get_body(uint32_t reference) {
				return collider_list[reference].body;
			}

			const inline fan_2d::body* get_body(uint32_t reference) const {
				return collider_list[reference].body;
			}

		protected:

			void BeginContact(b2Contact* contact) {
				b2Fixture *b2FixtureA = contact->GetFixtureA();
				b2Fixture *b2FixtureB = contact->GetFixtureB();
				b2Body* b2BodyA = b2FixtureA->GetBody();
				b2Body* b2BodyB = b2FixtureB->GetBody();
				
				uint32_t sprite_id_a = b2BodyA->GetUserData().pointer;
				uint32_t sprite_id_b = b2BodyB->GetUserData().pointer;

				if (collider_list[sprite_id_a].removed) {
					return;
				}
				if (collider_list[sprite_id_b].removed) {
					return;
				}

				if (m_on_collision) {
					m_on_collision((fan_2d::fixture*)contact->GetFixtureA(), (fan_2d::fixture*)contact->GetFixtureB());
				}	
			}

			void EndContact(b2Contact* contact) {

				b2Fixture *b2FixtureA = contact->GetFixtureA();
				b2Fixture *b2FixtureB = contact->GetFixtureB();
				b2Body* b2BodyA = b2FixtureA->GetBody();
				b2Body* b2BodyB = b2FixtureB->GetBody();

				uint32_t sprite_id_a = b2BodyA->GetUserData().pointer;
				uint32_t sprite_id_b = b2BodyB->GetUserData().pointer;

				if (collider_list[sprite_id_a].removed) {
					return;
				}
				if (collider_list[sprite_id_b].removed) {
					return;
				}

				if (m_on_collision_exit) {
					m_on_collision_exit((fan_2d::fixture*)contact->GetFixtureA(), (fan_2d::fixture*)contact->GetFixtureB());
				}
			}

			void PreSolve(b2Contact* contact, const b2Manifold* oldManifold) {

				b2Fixture *b2FixtureA = contact->GetFixtureA();
				b2Fixture *b2FixtureB = contact->GetFixtureB();
				b2Body* b2BodyA = b2FixtureA->GetBody();
				b2Body* b2BodyB = b2FixtureB->GetBody();

				uint32_t sprite_id_a = b2BodyA->GetUserData().pointer;
				uint32_t sprite_id_b = b2BodyB->GetUserData().pointer;

				if (collider_list[sprite_id_a].removed) {
					return;
				}
				if (collider_list[sprite_id_b].removed) {
					return;
				}

				if (m_on_presolve) {
					m_on_presolve((fan_2d::contact*)contact, (const fan_2d::manifold*)oldManifold);
				}
			}

		private:

			on_collision_t m_on_collision;
			on_collision_t m_on_collision_exit;
			on_presolve_t m_on_presolve;

		};

		template <typename user_struct>
		struct collider_wrap_t : public collider_callbacks_t<user_struct> {

			collider_wrap_t() { }

			collider_wrap_t(fan_2d::engine::engine_t* engine) : m_engine(engine) {
				m_engine->world.set_contact_listener(this);
			}

			// check if body should be woken

			void apply_force(uint32_t reference, const fan::vec2& force, const fan::vec2& point = 0) {
				collider_callbacks_t<user_struct>::get_body(reference)->apply_force(force, point);
			}

			void apply_impulse(uint32_t reference, const fan::vec2& force, const fan::vec2& point = 0) {
				collider_callbacks_t<user_struct>::get_body(reference)->apply_impulse(force, point);
			}

			void apply_torque(uint32_t reference, f32_t torque) {
				collider_callbacks_t<user_struct>::get_body(reference)->apply_torque(torque);
			}

			f32_t get_angular_damping(uint32_t reference) const {
				return collider_callbacks_t<user_struct>::get_body(reference)->get_angular_damping();
			}
			void set_angular_damping(uint32_t reference, f32_t amount) {
				collider_callbacks_t<user_struct>::get_body(reference)->set_angular_damping(amount);
			}

			f32_t get_angular_velocity(uint32_t reference) const {
				return collider_callbacks_t<user_struct>::get_body(reference)->get_angular_velocity();
			}
			void set_angular_velocity(uint32_t reference, f32_t velocity) {
				collider_callbacks_t<user_struct>::get_body(reference)->set_angular_velocity(velocity);
			}

			fan_2d::fixture* get_fixture_list(uint32_t reference) {
				return (fan_2d::fixture*)collider_callbacks_t<user_struct>::get_body(reference)->get_fixture_list();
			}

			fan_2d::fixture* get_next(uint32_t reference) {
				return (fan_2d::fixture*)collider_callbacks_t<user_struct>::get_body(reference)->get_next();
			}

			f32_t get_gravity_scale(uint32_t reference) const {
				return collider_callbacks_t<user_struct>::get_body(reference)->get_gravity_scale();
			}
			void set_gravity_scale(uint32_t reference, f32_t scale) {
				collider_callbacks_t<user_struct>::get_body(reference)->set_gravity_scale(scale);
			}

			f32_t get_velocity_damping(uint32_t reference) const {
				return collider_callbacks_t<user_struct>::get_body(reference)->get_velocity_damping();
			}
			void set_velocity_damping(uint32_t reference, f32_t amount) {
				collider_callbacks_t<user_struct>::get_body(reference)->set_velocity_damping(amount);
			}

			fan::vec2 get_velocity(uint32_t reference) const {
				return collider_callbacks_t<user_struct>::get_body(reference)->get_velocity();
			}
			void set_velocity(uint32_t reference, const fan::vec2& velocity) {
				collider_callbacks_t<user_struct>::get_body(reference)->set_velocity(velocity);
			}

			body_type_t get_body_type(uint32_t reference) const {
				return (body_type_t)collider_callbacks_t<user_struct>::get_body(reference)->get_body_type();
			}

			void set_fixed_rotation(uint32_t reference, bool flag) {
				collider_callbacks_t<user_struct>::get_body(reference)->set_fixed_rotation(flag);
			}

		/*	f32_t get_density(uint32_t reference) {
				return physics_base::physics_callbacks::m_fixture[reference]->get_density();
			}
			void set_density(uint32_t reference, f32_t density) {
				physics_base::physics_callbacks::m_fixture[reference]->set_density(density);
			}*/

			f32_t get_mass(uint32_t reference) {
				return collider_callbacks_t<user_struct>::get_body(reference)->get_mass();
			}

			bool is_bullet(uint32_t reference) const {
				return collider_callbacks_t<user_struct>::get_body(reference)->is_bullet();
			}

			bool is_awake(uint32_t reference) const {
				return collider_callbacks_t<user_struct>::get_body(reference)->is_awake();
			}
			void set_awake(uint32_t reference, bool flag) {
				return collider_callbacks_t<user_struct>::get_body(reference)->set_awake(flag);
			}

			bool is_enabled(uint32_t reference) const {
				return collider_callbacks_t<user_struct>::get_body(reference)->is_enabled();
			}
			void set_enabled(uint32_t reference, bool flag) {
				collider_callbacks_t<user_struct>::get_body(reference)->set_enabled(flag);
			}

			fan::vec2 get_position(uint32_t reference) const {
				return collider_callbacks_t<user_struct>::get_body(reference)->get_position();
			}
			void set_position(uint32_t reference, const fan::vec2& position) {
				collider_callbacks_t<user_struct>::get_body(reference)->set_transform((position).b2(), collider_callbacks_t<user_struct>::get_body(reference)->get_angle());
				collider_callbacks_t<user_struct>::get_body(reference)->set_awake(true);
			}

			f32_t get_angle(uint32_t reference) const {
				return collider_callbacks_t<user_struct>::get_body(reference)->get_angle();
			}
			void set_angle(uint32_t reference, f64_t rotation) {
				collider_callbacks_t<user_struct>::get_body(reference)->set_transform(collider_callbacks_t<user_struct>::get_body(reference)->get_position(), -rotation);
			}

			void set_angular_rotation(uint32_t reference, f64_t w) {
				collider_callbacks_t<user_struct>::get_body(reference)->set_angular_velocity(w);
				collider_callbacks_t<user_struct>::get_body(reference)->set_awake(true);
			}

			void set_sensor(uint32_t reference, bool flag) {
				auto ptr = collider_callbacks_t<user_struct>::get_body(reference)->get_fixture_list();

					for (; ptr; ptr = ptr->get_next()) {
						ptr->set_sensor(flag);
					}
				//collider_callbacks_t<user_struct>::get_body(reference)->set_sensor(flag);
			}

			void erase(uint32_t reference) {

				this->collider_list[reference].removed = 1;

				auto f = [r = reference, this] {

					auto ptr = collider_callbacks_t<user_struct>::get_body(r)->get_fixture_list();

					while (ptr) {
						auto ptr_next = ptr->get_next();
						collider_callbacks_t<user_struct>::get_body(r)->destroy_fixture(ptr);
						ptr = ptr_next;
					}

					m_engine->world.destroy_body(collider_callbacks_t<user_struct>::get_body(r));
					collider_callbacks_t<user_struct>::collider_list.unlink(r);
				};

				if (m_engine->world.is_locked()) {
					m_engine->m_queue_after_step.emplace_back(f);
				}
				else {
					f();
				}
					
			}

		protected:

			fan_2d::engine::engine_t* m_engine;

		};

		struct empty {

		};

		template <typename user_struct = empty>
		struct collider_t : public collider_wrap_t<user_struct> {

			using collider_wrap_t<user_struct>::collider_wrap_t;

		protected:
			//fan_2d::physics::triangle* physics_rectangle;

		public:

			user_struct get_user_data(uint32_t collider_reference) const {
				return collider_wrap_t<user_struct>::collider_list[collider_reference].user_data;
			}
			void set_user_data(uint32_t collider_reference, user_struct user_data) {
				collider_wrap_t<user_struct>::collider_list[collider_reference].user_data = user_data;
			}

			uint32_t size() const {
				return collider_wrap_t<user_struct>::collider_list.size();
			}

			uint32_t push_back(user_struct& data) {

				typename fan_2d::physics::collider_callbacks_t<user_struct>::node_t node;
				node.body = nullptr;
				node.user_data = data;

				return collider_wrap_t<user_struct>::collider_list.push_back(node);
			}

			void add_rectangle(uint32_t collider_reference, fan_2d::physics::rectangle_t::properties_t properties) {
				
				auto f = [&, address = (uint64_t)collider_wrap_t<user_struct>::m_engine, properties_ = properties] {

					auto* collider = &collider_wrap_t<user_struct>::collider_list[collider_reference];

					if (collider->body == nullptr) {

						fan_2d::body_def body_def;
						body_def.type = (b2BodyType)properties_.body_type;
						body_def.position.Set((properties_.position.x) / meter_scale, (properties_.position.y) / meter_scale);
						body_def.angle = -properties_.angle;

						fan_2d::engine::engine_t* engine = (fan_2d::engine::engine_t*)address;

						collider->body = engine->world.create_body(&body_def);
						collider->body->set_user_data(collider_reference);
					}

					collider->removed = false;

					b2PolygonShape shape;

					shape.SetAsBox((properties_.size.x) / meter_scale, (properties_.size.y) / meter_scale, (((properties_.position) / meter_scale - properties_.position / meter_scale)).b2(), -properties_.angle);

					b2FixtureDef fixture_def;
					fixture_def.shape = &shape;
					fixture_def.density = properties.body_property.mass;
					fixture_def.friction = properties.body_property.friction;
					fixture_def.restitution = properties.body_property.restitution;

					auto ptr = collider->body->create_fixture(&fixture_def);
					ptr->set_shape_type(fan_2d::shape_type::rectangle);
				};

				if (collider_wrap_t<user_struct>::m_engine->world.is_locked()) {
					collider_wrap_t<user_struct>::m_engine->m_queue_after_step.emplace_back(f);
				}
				else{
					f();
				}
			}
			
			//void physics_add_rectangle(uint32_t SpriteID, fan_2d::physics::rectangle::properties_t properties) {

			//	if (inherit_t::m_engine->world.is_locked()) {
			//		auto f = [&, SpriteID_ = SpriteID, properties = properties] () mutable {
			//			typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(SpriteID_);
			//			auto sprite_node = &node->data;
			//			if (sprite_node->physics_nodereference == (uint32_t)-1) {
			//				sprite_node->physics_nodereference = inherit_t::m_body.new_node_last();
			//				inherit_t::m_body[sprite_node->physics_nodereference] = nullptr;
			//			}
			//			else if (sprite_node->removed == true) {
			//				sprite_node->removed = false;
			//			}

			//			if (sprite_node->graphics_nodereference != (uint32_t)-1) {
			//				properties.angle += fan_2d::opengl::sprite_t::get_angle(sprite_node->graphics_nodereference);
			//				fan_2d::opengl::sprite_t::set_angle(sprite_node->graphics_nodereference, properties.angle);
			//			}

			//			inherit_t::physics_add_rectangle(properties.position, sprite_node->physics_nodereference, SpriteID_, properties);
			//			//sprite_node->physics_nodereference = physics_size();
			//			//if (sprite_node->graphics_nodereference != (uint32_t)-1) { // ?
			//			//	inherit_t::sprite::physics_add_rectangle(inherit_t::sprite::get_position(sprite_node->graphics_nodereference) + inherit_t::sprite::get_size(sprite_node->graphics_nodereference) / 2, sprite_node->physics_nodereference, SpriteID, properties);
			//			//}
			//			//else {
			//			//	
			//			//}
			//		};
			//		inherit_t::m_engine->m_queue_after_step.emplace_back(f);
			//	}
			//	else {
			//		typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(SpriteID);
			//		auto sprite_node = &node->data;
			//		if (sprite_node->physics_nodereference == (uint32_t)-1) {
			//			sprite_node->physics_nodereference = inherit_t::m_body.new_node_last();
			//			inherit_t::m_body[sprite_node->physics_nodereference] = nullptr;
			//		}
			//		if (sprite_node->graphics_nodereference != (uint32_t)-1) {
			//			properties.angle += fan_2d::opengl::sprite_t::get_angle(sprite_node->graphics_nodereference);
			//			fan_2d::opengl::sprite_t::set_angle(sprite_node->graphics_nodereference, properties.angle);
			//		}
			//		if (sprite_node->graphics_nodereference != (uint32_t)-1) {
			//			fan_2d::physics::sprite<user_struct_t>::physics_add_rectangle(fan_2d::opengl::sprite_t::get_position(sprite_node->graphics_nodereference), sprite_node->physics_nodereference, SpriteID, properties);
			//		}
			//		else {
			//			fan_2d::physics::sprite<user_struct_t>::physics_add_rectangle(properties.position, sprite_node->physics_nodereference, SpriteID, properties);
			//		}
			//	}
			//}
			void add_circle(uint32_t collider_reference, const fan_2d::physics::circle_t::properties_t& properties) {

				auto* collider = &collider_wrap_t<user_struct>::collider_list[collider_reference];

				auto f = [&, address = (uint64_t)collider_wrap_t<user_struct>::m_engine, properties_ = properties] {

					if (collider->body == nullptr) {
						fan_2d::body_def body_def;
						body_def.type = (b2BodyType)properties_.body_type;
						body_def.position.Set(properties.position.x / meter_scale, properties.position.y / meter_scale);

						fan_2d::engine::engine_t* engine = (fan_2d::engine::engine_t*)address;

						collider->body = engine->world.create_body(&body_def);
						collider->body->set_user_data(collider_reference);
					}

					collider->removed = false;

					b2CircleShape shape;
					fan::vec2 bp = collider->body->get_position();
					fan::vec2 np((properties.position - bp) / meter_scale);
					shape.m_p.Set(np.x, np.y);
					shape.m_radius = properties.radius / meter_scale;

					b2FixtureDef fixture_def;
					fixture_def.shape = &shape;
					fixture_def.density = properties.body_property.mass;
					fixture_def.friction = properties.body_property.friction;

					auto ptr = (fan_2d::fixture*)collider->body->create_fixture(&fixture_def);
					ptr->set_shape_type(fan_2d::shape_type::circle);
				};

				if (collider_wrap_t<user_struct>::m_engine->world.is_locked()) {
					collider_wrap_t<user_struct>::m_engine->m_queue_after_step.emplace_back(f);
				}
				else{
					f();
				}

			}

			/*void physics_add_circle(uint32_t physics_id, uint32_t sprite_id, const fan_2d::physics::circle::properties_t& properties) {

			if (!inherit_t::m_body[physics_id]) {
					fan_2d::body_def body_def;
					body_def.type = (b2BodyType)properties.body_type;
					body_def.position.Set(properties.position.x / meter_scale, properties.position.y / meter_scale);
					inherit_t::m_body[physics_id] = inherit_t::m_world->create_body(&body_def);
					inherit_t::m_body[physics_id]->set_user_data(sprite_id);
				}

				b2CircleShape shape;
				fan::vec2 bp = inherit_t::m_body[physics_id]->get_position();
				fan::vec2 np((properties.position - bp) / meter_scale);
				shape.m_p.Set(np.x, np.y);
				shape.m_radius = properties.radius / meter_scale;

				b2FixtureDef fixture_def;
				fixture_def.shape = &shape;
				fixture_def.density = properties.body_property.mass;
				fixture_def.friction = properties.body_property.friction;

				auto ptr = ((fan_2d::fixture*)inherit_t::m_body[physics_id]->create_fixture(&fixture_def));
				ptr->set_shape_type(fan_2d::shape_type::circle);
			}*/

		};

		/*struct motor_joint : public physics_base<std::vector<fan_2d::fixture*>, std::vector<fan_2d::body*>> {

			motor_joint(fan_2d::world* world) : collider_wrap_t(world) {}

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

				wheel_joints.emplace_back((fan_2d::wheel_joint*)m_world->create_joint(&jd));
			}

			void erase(uintptr_t i) {
				collider_wrap_t::erase(i);
				m_world->destroy_joint(wheel_joints[i]);
				wheel_joints.erase(wheel_joints.begin() + i);
			}
			void erase(uintptr_t begin, uintptr_t end) {

				collider_wrap_t::erase(begin, end);

				for (int i = begin; i < end; i++) {
					m_world->destroy_joint(wheel_joints[i]);
				}

				wheel_joints.erase(wheel_joints.begin() + begin, wheel_joints.begin() + end);
			}

			auto get_body_a(uint32_t reference) {
				return wheel_joints[i]->GetBodyA();
			}

			auto get_body_b(uint32_t reference) {
				return wheel_joints[i]->GetBodyB();
			}

			void set_speed(uint32_t reference, f32_t speed) {
				wheel_joints[i]->SetMotorSpeed(speed);
			}

			auto size() const {
				return wheel_joints.size();
			}

		protected:

			using collider_wrap_t::get_body;

			std::vector<fan_2d::wheel_joint*> wheel_joints;

		};*/

	};

}

#pragma once

#include <fan/graphics/graphics.h>

#include <fan/bll.h>

namespace fan_2d {

	namespace engine {

		template <typename graphics_t, typename physics_t>
		struct base_engine : public graphics_t, public physics_t {

			base_engine(fan_2d::engine::engine_t* engine) : m_engine(engine), graphics_t(&engine->context.camera), physics_t(&engine->world) { }

			void set_rotation(uintptr_t i, f_t angle) {

				graphics_t::set_angle(i, angle);

				physics_t::set_angle(i, angle);

			}

			void erase(uintptr_t i) {
				std::function<void()> f = [&, i_ = i] {
					graphics_t::erase(i_);
					physics_t::erase(i_);
				};

				m_engine->m_queue_after_step.emplace_back(f);
			}
			void erase(uintptr_t begin, uintptr_t end) {
				std::function<void()> f = [&, begin_ = begin, end_ = end] {
					graphics_t::erase(begin_, end_);
					physics_t::erase(begin_, end_);
				};

				m_engine->m_queue_after_step.emplace_back(f);
			}

			fan_2d::engine::engine_t* m_engine;

		};

	/*	struct rectangle : public base_engine<fan_2d::opengl::rectangle, fan_2d::physics::rectangle> {

			rectangle(fan_2d::engine::engine_t* engine) : rectangle::base_engine(engine) {
				engine->push_update(this, [&] { rectangle::update_position(); });
			}

			struct properties_t : public fan_2d::opengl::rectangle::properties_t {
				fan_2d::physics::body_type_t body_type;
				fan_2d::physics::body_property_t body_property = { 10, 1, 0.1 };		
			};

			void push_back(properties_t properties) {

				properties.rotation_point = properties.size / 2;

				fan_2d::opengl::rectangle::push_back(properties);

				fan_2d::physics::rectangle::properties_t p_property;
				p_property.position = properties.position / meter_scale;
				p_property.size = properties.size / meter_scale;
				p_property.angle = properties.angle;
				p_property.body_property = properties.body_property;
				p_property.body_type = properties.body_type;

				fan_2d::physics::rectangle::push_back(p_property);
			}

			void set_position(uintptr_t i, const fan::vec2& position) {
				fan_2d::opengl::rectangle::set_position(i, (position - get_size(i) * 0.5) * meter_scale);
				fan_2d::physics::rectangle::set_position(i, position);
			}

			void update_position() {

				for (int i = 0; i < this->size(); i++) {
					const fan::vec2 new_position = this->get_body(i)->get_position() - fan_2d::opengl::rectangle::get_size(i) * 0.5;

					const f32_t new_angle = -this->get_body(i)->get_angle();

					if (this->get_body(i)->get_body_type() == fan_2d::physics::body_type_t::static_body ||
						fan_2d::opengl::rectangle::get_position(i) == new_position
						&& fan_2d::opengl::rectangle::get_angle(i) == new_angle) {
						continue;
					}

					fan_2d::opengl::rectangle::set_position(i, new_position);
					fan_2d::opengl::rectangle::set_angle(i, new_angle);
				}
			}

		};*/

	/*	template <typename user_struct_t = fan_2d::physics::empty>
		struct base_sprite_t : public base_engine<fan_2d::opengl::sprite_t, fan_2d::physics::sprite<user_struct_t>> {

		private:

			using fan_2d::opengl::sprite_t::push_back;
			using fan_2d::physics::sprite<user_struct_t>::physics_size;

		public:

			using base_sprite_inherit_t = base_engine<fan_2d::opengl::sprite_t, fan_2d::physics::sprite<user_struct_t>>;

			using inherit_t = base_sprite_inherit_t;

			struct properties_t : public fan_2d::opengl::sprite_t::properties_t {

				fan_2d::physics::body_type_t body_type;
				fan_2d::physics::body_property_t body_property = { 10, 1, 0.1 };		
			};

			base_sprite_t(fan_2d::engine::engine_t* engine) : base_sprite_t::base_engine(engine) {
				engine->push_update(this, [&] { base_sprite_t::update_position(); });
			}

			void erase_graphics(uint32_t sprite_id) {
				auto& ref = base_sprite_inherit_t::sprite_list[sprite_id].graphics_nodereference;
				fan_2d::opengl::sprite_t::erase(ref);
				uint32_t old_ref = ref;
				ref = -1;

				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(base_sprite_inherit_t::sprite_list.begin());

				while (node->next != base_sprite_inherit_t::sprite_list.dst) {
						if (base_sprite_inherit_t::sprite_list[node->next].graphics_nodereference != (uint32_t)-1 && base_sprite_inherit_t::sprite_list[node->next].graphics_nodereference > old_ref) {
							base_sprite_inherit_t::sprite_list[node->next].graphics_nodereference--;
						}
						node = base_sprite_inherit_t::sprite_list.get_node_by_reference(node->next);
					}
			}

			void erase_physics(uint32_t sprite_id) {

				base_sprite_inherit_t::sprite_list[sprite_id].removed = 1;

				if (inherit_t::m_engine->world.is_locked()) {
					auto f = [&, sprite_id_ = sprite_id] {
						fan_2d::physics::sprite<user_struct_t>::erase(sprite_id_, base_sprite_inherit_t::sprite_list[sprite_id_].physics_nodereference);
					};
					inherit_t::m_engine->m_queue_after_step.emplace_back(f);
				}
				else {
					fan_2d::physics::sprite<user_struct_t>::erase(sprite_id, base_sprite_inherit_t::sprite_list[sprite_id].physics_nodereference);
				}

			}

			fan::vec2 get_position(uint32_t sprite_id) const {
				auto node = base_sprite_inherit_t::sprite_list[sprite_id];
				if (node.graphics_nodereference < fan_2d::opengl::sprite_t::size() && node.graphics_nodereference != (uint32_t)-1) {
					return fan_2d::opengl::sprite_t::get_position(node.graphics_nodereference);
				}
				else if (node.physics_nodereference < fan_2d::physics::sprite<user_struct_t>::size() && node.physics_nodereference != (uint32_t)-1) {
					return fan_2d::physics::sprite<user_struct_t>::get_position(node.physics_nodereference);
				}
				else {
					throw std::runtime_error("corrupted sprite id xd");
				}
			}

			void set_position(uint32_t sprite_id, const fan::vec2& position) {
				if (base_sprite_inherit_t::sprite_list[sprite_id].physics_nodereference != -1) {
					throw std::runtime_error("cannot have physics while setting position");
				}

				fan_2d::opengl::sprite_t::set_position(base_sprite_inherit_t::sprite_list[sprite_id].graphics_nodereference, position);
			}

			fan::vec2 get_size(uint32_t sprite_id) const {
				return fan_2d::opengl::sprite_t::get_size(base_sprite_inherit_t::sprite_list[sprite_id].graphics_nodereference);
			}

			f32_t get_transparency(uint32_t sprite_id) const {
				return fan_2d::opengl::sprite_t::get_transparency(base_sprite_inherit_t::sprite_list[sprite_id].graphics_nodereference);
			}
			void set_transparency(uint32_t sprite_id, f32_t transparency) {
				return fan_2d::opengl::sprite_t::set_transparency(base_sprite_inherit_t::sprite_list[sprite_id].graphics_nodereference, transparency);
			}

			f32_t get_angle(uint32_t sprite_id) const {
				return fan_2d::opengl::sprite_t::get_angle(base_sprite_inherit_t::sprite_list[sprite_id].graphics_nodereference);
			}

			void apply_force(uint32_t sprite_id, const fan::vec2& force, const fan::vec2& point = 0) {

				if (inherit_t::m_engine->world.is_locked()) {
					auto f = [&, id = sprite_id, force_ = force, point_ = point] {
						typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(id);
						inherit_t::m_body[node->data.physics_nodereference]->apply_force(force_, point_);
					};

					inherit_t::m_engine->m_queue_after_step.emplace_back(f);
				}
				else {
					typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
					inherit_t::m_body[node->data.physics_nodereference]->apply_force(force, point);
				}

			}

			void apply_impulse(uint32_t sprite_id, const fan::vec2& force, const fan::vec2& point = 0) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->apply_impulse(force, point);
			}

			void apply_torque(uint32_t sprite_id, f32_t torque) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->apply_torque(torque);
			}

			f32_t get_angular_damping(uint32_t sprite_id) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_angular_damping();
			}
			void set_angular_damping(uint32_t sprite_id, f32_t amount) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_angular_damping(amount);
			}

			f32_t get_angular_velocity(uint32_t sprite_id) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_angular_velocity();
			}
			void set_angular_velocity(uint32_t sprite_id, f32_t velocity) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_angular_velocity(velocity);
			}

			fan_2d::fixture* get_fixture_list(uint32_t sprite_id) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return (fan_2d::fixture*)inherit_t::m_body[node->data.physics_nodereference]->get_fixture_list();
			}

			f32_t get_gravity_scale(uint32_t sprite_id) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_gravity_scale();
			}
			void set_gravity_scale(uint32_t sprite_id, f32_t scale) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_gravity_scale(scale);
			}

			f32_t get_velocity_damping(uint32_t sprite_id) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_velocity_damping();
			}
			void set_velocity_damping(uint32_t sprite_id, f32_t amount) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_velocity_damping(amount);
			}

			bool is_fixed_rotation(uint32_t sprite_id) const {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return this->inherit_t::m_body[node->data.physics_nodereference]->is_fixed_rotation();
			}
			void set_fixed_rotation(uint32_t sprite_id, bool flag) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				this->inherit_t::m_body[node->data.physics_nodereference]->set_fixed_rotation(flag);
			}

			fan::vec2 get_velocity(uint32_t sprite_id) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_velocity();
			}
			void set_velocity(uint32_t sprite_id, const fan::vec2& velocity) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_velocity(velocity);
			}

			fan_2d::physics::body_type_t get_body_type(uint32_t sprite_id) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return (fan_2d::physics::body_type_t)inherit_t::m_body[node->data.physics_nodereference]->get_body_type();
			}

			f32_t get_density(uint32_t sprite_id) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_fixture[node->data.physics_nodereference]->get_density();
			}
			void set_density(uint32_t sprite_id, f32_t density) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_fixture[node->data.physics_nodereference]->set_density(density);
			}

			f32_t get_mass(uint32_t sprite_id) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_mass();
			}

			bool is_bullet(uint32_t sprite_id) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->is_bullet();
			}

			bool is_awake(uint32_t sprite_id) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->is_awake();
			}
			void set_awake(uint32_t sprite_id, bool flag) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->set_awake(flag);
			}

			bool is_enabled(uint32_t sprite_id) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->is_enabled();
			}
			void set_enabled(uint32_t sprite_id, bool flag) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_enabled(flag);
			}

			f32_t get_restitution(uint32_t sprite_id) const {
				const typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_fixture[node->data.physics_nodereference]->get_restitution();
			}
			void set_restitution(uint32_t sprite_id, f32_t restitution) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_fixture[node->data.physics_nodereference]->set_restitution(restitution);
			}

			void set_angle(uint32_t sprite_id, f64_t rotation) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				this->inherit_t::m_body[node->data.physics_nodereference]->set_transform(this->inherit_t::m_body[node->data.physics_nodereference]->get_position(), -rotation);
			}

			void set_angular_rotation(uint32_t sprite_id, f64_t w) {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				this->inherit_t::m_body[node->data.physics_nodereference]->set_angular_velocity(w);
				this->inherit_t::m_body[node->data.physics_nodereference]->set_awake(true);
			}

			fan::vec2 get_world_center(uint32_t sprite_id) const {
				typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
				return this->inherit_t::m_body[node->data.physics_nodereference]->get_world_center();
			}

			void set_sensor(uint32_t sprite_id, bool flag) {
				if (inherit_t::m_engine->world.is_locked()) {
					auto f = [&, id = sprite_id, flag_ = flag] {
						typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(id);
						
						auto ptr = inherit_t::m_body[node->data.physics_nodereference]->get_fixture_list();

						for (; ptr; ptr = ptr->get_next()) {
							ptr->set_sensor(flag_);
						}
					};

					inherit_t::m_engine->m_queue_after_step.emplace_back(f);
				}
				else {
					typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);

					auto ptr = inherit_t::m_body[node->data.physics_nodereference]->get_fixture_list();

					for (; ptr; ptr = ptr->get_next()) {
						ptr->set_sensor(flag);
					}
				}

			}

			using sprite = base_sprite_t<>;*/


			/*void erase(uintptr_t i) {
			decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);
			m_world->DestroyBody((b2Body*)inherit_t::get_body(i));

			inherit_t::m_fixture.erase(inherit_t::m_fixture.begin() + i);
			inherit_t::m_body.erase(inherit_t::m_body.begin() + i);

			}*/
			/*void erase(uintptr_t begin, uintptr_t end) {
			decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id);

			if (begin > inherit_t::m_body.size() || end > inherit_t::m_body.size()) {
			return;
			}

			for (int i = begin; i < end; i++) {
			m_world->DestroyBody((b2Body*)inherit_t::get_body(i));
			}

			inherit_t::m_fixture.erase(inherit_t::m_fixture.begin() + begin, inherit_t::m_fixture.begin() + end);
			inherit_t::m_body.erase(inherit_t::m_body.begin() + begin, inherit_t::m_body.begin() + end);
			}*/

			/*void push_back(fan_2d::opengl::sprite_t::properties_t properties) {

			}*/

			//void update_position() {

			//	for (int i = base_sprite_inherit_t::sprite_list.get_node_first(); i != base_sprite_inherit_t::sprite_list.dst; ) {

			//		if (base_sprite_inherit_t::sprite_list[i].physics_nodereference == (uint32_t)-1) {
			//			typename decltype(base_sprite_inherit_t::sprite_list)::node_t *node = base_sprite_inherit_t::sprite_list.get_node_by_reference(i);
			//			i = node->next;
			//			continue;
			//		}


			//		fan::vec2 new_position = this->m_body[base_sprite_inherit_t::sprite_list[i].physics_nodereference]->get_position();

			//		f32_t new_angle = -this->m_body[base_sprite_inherit_t::sprite_list[i].physics_nodereference]->get_angle();

			//		fan::vec2 old_angle = fan_2d::opengl::sprite_t::get_angle(base_sprite_inherit_t::sprite_list[i].graphics_nodereference);
			//		fan::vec2 old_position = fan_2d::opengl::sprite_t::get_position(base_sprite_inherit_t::sprite_list[i].graphics_nodereference);

			//		if (new_position == old_position && new_angle == old_angle) {
			//			typename decltype(base_sprite_inherit_t::sprite_list)::node_t *node = base_sprite_inherit_t::sprite_list.get_node_by_reference(i);
			//			i = node->next;
			//			continue;
			//		}

			//		fan_2d::opengl::sprite_t::set_angle(base_sprite_inherit_t::sprite_list[i].graphics_nodereference, new_angle);
			//		fan_2d::opengl::sprite_t::set_position(base_sprite_inherit_t::sprite_list[i].graphics_nodereference, new_position);

			//		typename decltype(base_sprite_inherit_t::sprite_list)::node_t *node = base_sprite_inherit_t::sprite_list.get_node_by_reference(i);
			//		i = node->next;
			//	}
			//}

			//void sprite_init_graphics(uint32_t nodereference, const fan_2d::opengl::sprite_t::properties_t& properties) {

			//	typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(nodereference);
			//	auto sprite_node = &node->data;
			//	if (sprite_node->graphics_nodereference != -1) {
			//		// already exists
			//		assert(0);
			//	}

			//	sprite_node->graphics_nodereference = fan_2d::opengl::sprite_t::size();

			//	fan_2d::opengl::sprite_t::push_back(properties);
			//}
			//void physics_add_triangle(uint32_t SpriteID, const fan_2d::physics::triangle::properties_t& properties) {

			//	if (inherit_t::m_engine->world.is_locked()) {

			//		auto f = [&, SpriteID_ = SpriteID, properties = properties]{

			//			typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(SpriteID_);
			//			auto sprite_node = &node->data;
			//			if (sprite_node->physics_nodereference == (uint32_t)-1) {
			//				sprite_node->physics_nodereference = inherit_t::m_body.new_node_last();
			//				inherit_t::m_body[sprite_node->physics_nodereference] = nullptr;
			//			}
			//			else if (sprite_node->removed == true) {
			//				sprite_node->removed = false;
			//			}

			//			if (sprite_node->graphics_nodereference != (uint32_t)-1) {
			//				properties.angle += fan_2d::opengl::sprite_t::get_angle(sprite_node->graphics_nodereference);
			//				fan_2d::opengl::sprite_t::set_angle(sprite_node->graphics_nodereference, properties.angle);
			//			}

			//			fan_2d::physics::sprite<user_struct_t>::physics_add_triangle(sprite_node->physics_nodereference, SpriteID_, properties);
			//		};

			//		inherit_t::m_engine->m_queue_after_step.emplace_back(f);
			//	}
			//	else {

			//		typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(SpriteID);
			//		auto sprite_node = &node->data;
			//		if (sprite_node->physics_nodereference == (uint32_t)-1) {
			//			sprite_node->physics_nodereference = inherit_t::m_body.new_node_last();
			//			inherit_t::m_body[sprite_node->physics_nodereference] = nullptr;
			//		}
			//		else if (sprite_node->removed == true) {
			//			sprite_node->removed = false;
			//		}

			//		if (sprite_node->graphics_nodereference != (uint32_t)-1) {
			//			properties.angle += fan_2d::opengl::sprite_t::get_angle(sprite_node->graphics_nodereference);
			//			fan_2d::opengl::sprite_t::set_angle(sprite_node->graphics_nodereference, properties.angle);
			//		}

			//		fan_2d::physics::sprite<user_struct_t>::physics_add_triangle(sprite_node->physics_nodereference, SpriteID, properties);
			//	}

			//}
			//

			//void physics_add_convex(uint32_t SpriteID, const fan_2d::physics::convex::properties_t& properties) {

			//	if (inherit_t::m_engine->world.is_locked()) {
			//		auto f = [this, SpriteID_ = SpriteID, properties = properties] {
			//			typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = inherit_t::sprite_list.get_node_by_reference(SpriteID_);
			//			auto sprite_node = &node->data;
			//			if (sprite_node->physics_nodereference == (uint32_t)-1) {
			//				sprite_node->physics_nodereference = inherit_t::m_body.new_node_last();
			//				inherit_t::m_body[sprite_node->physics_nodereference] = nullptr;
			//			}
			//			else if (sprite_node->removed == true) {
			//				sprite_node->removed = false;
			//			}

			//			if (sprite_node->graphics_nodereference != (uint32_t)-1) {
			//				properties.angle += fan_2d::opengl::sprite_t::get_angle(sprite_node->graphics_nodereference);
			//				fan_2d::opengl::sprite_t::set_angle(sprite_node->graphics_nodereference, properties.angle);
			//			}

			//			fan_2d::physics::sprite<user_struct_t>::physics_add_convex(sprite_node->physics_nodereference, SpriteID_, properties);
			//		};

			//		inherit_t::m_engine->m_queue_after_step.emplace_back(f);
			//	}
			//	else {
			//		typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(SpriteID);
			//		auto sprite_node = &node->data;
			//		if (sprite_node->physics_nodereference == (uint32_t)-1) {
			//			sprite_node->physics_nodereference = inherit_t::m_body.new_node_last();
			//			inherit_t::m_body[sprite_node->physics_nodereference] = nullptr;
			//		}
			//		else if (sprite_node->removed == true) {
			//			sprite_node->removed = false;
			//		}

			//		if (sprite_node->graphics_nodereference != (uint32_t)-1) {
			//			properties.angle += fan_2d::opengl::sprite_t::get_angle(sprite_node->graphics_nodereference);
			//			fan_2d::opengl::sprite_t::set_angle(sprite_node->graphics_nodereference, properties.angle);
			//		}

			//		fan_2d::physics::sprite<user_struct_t>::physics_add_convex(sprite_node->physics_nodereference, SpriteID, properties);
			//	}

			//}

			//void physics_add_edge(uint32_t SpriteID, const fan_2d::physics::edge_shape::properties_t& properties) {

			//	if (inherit_t::m_engine->world.is_locked()) {
			//		auto f = [this, SpriteID_ = SpriteID, properties = properties] {
			//			typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = inherit_t::sprite_list.get_node_by_reference(SpriteID_);
			//			auto sprite_node = &node->data;
			//			if (sprite_node->physics_nodereference == (uint32_t)-1) {
			//				sprite_node->physics_nodereference = inherit_t::m_body.new_node_last();
			//				inherit_t::m_body[sprite_node->physics_nodereference] = nullptr;
			//			}
			//			else if (sprite_node->removed == true) {
			//				sprite_node->removed = false;
			//			}

			//			if (sprite_node->graphics_nodereference != (uint32_t)-1) {
			//				properties.angle += fan_2d::opengl::sprite_t::get_angle(sprite_node->graphics_nodereference);
			//				fan_2d::opengl::sprite_t::set_angle(sprite_node->graphics_nodereference, properties.angle);
			//			}

			//			fan_2d::physics::sprite<user_struct_t>::physics_add_edge(sprite_node->physics_nodereference, SpriteID_, properties);
			//		};

			//		inherit_t::m_engine->m_queue_after_step.emplace_back(f);
			//	}
			//	else {
			//		typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(SpriteID);
			//		auto sprite_node = &node->data;
			//		if (sprite_node->physics_nodereference == (uint32_t)-1) {
			//			sprite_node->physics_nodereference = inherit_t::m_body.new_node_last();
			//			inherit_t::m_body[sprite_node->physics_nodereference] = nullptr;
			//		}
			//		else if (sprite_node->removed == true) {
			//			sprite_node->removed = false;
			//		}

			//		if (sprite_node->graphics_nodereference != (uint32_t)-1) {
			//			properties.angle += fan_2d::opengl::sprite_t::get_angle(sprite_node->graphics_nodereference);
			//			fan_2d::opengl::sprite_t::set_angle(sprite_node->graphics_nodereference, properties.angle);
			//		}

			//		fan_2d::physics::sprite<user_struct_t>::physics_add_edge(sprite_node->physics_nodereference, SpriteID, properties);
			//	}

			//}

			//void sprite_init_physics(uint32_t nodereference) {
			//	typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(nodereference);
			//	auto sprite_node = &node->data;
			//	if (sprite_node->physics_nodereference != (uint32_t)-1) {
			//		// already exists
			//		assert(0);
			//	}
			//	sprite_node->physics_nodereference = physics_size();

			//	// push_back things here
			//}
			////void sprite_init_physics(uint32_t nodereference) {
			////	decltype(bll)::node_t* node = bll.get_node_by_reference(nodereference);
			////	sprite_node_t<user_struct_t>* sprite_node = &node->data;
			////	if (sprite_node->physics_nodereference != -1) {
			////		// already exists
			////		assert(0);
			////	}
			////	sprite_node->physics_nodereference = physics_size();

			////	// push_back things here
			////}

			//const user_struct_t get_user_data(uint32_t sprite_id) const {
			//	return base_sprite_inherit_t::sprite_list[sprite_id].user_data;
			//}

			//user_struct_t& get_user_data(uint32_t sprite_id) {
			//	return base_sprite_inherit_t::sprite_list[sprite_id].user_data;
			//}

			//uint32_t push_sprite(user_struct_t user_struct) {
			//	uint32_t nodereference = base_sprite_inherit_t::sprite_list.new_node_last();
			//	typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(nodereference);
			//	auto sprite_node = &node->data;
			//	sprite_node->removed = 0;
			//	sprite_node->user_data = user_struct;
			//	sprite_node->graphics_nodereference = -1;
			//	sprite_node->physics_nodereference = -1;

			//	return nodereference;
			//}
			//void erase(uint32_t sprite_id) {

			//	base_sprite_inherit_t::sprite_list[sprite_id].removed = 1;

			//	auto f = [&, sprite_id_ = sprite_id] {

			//		typename decltype(base_sprite_inherit_t::sprite_list)::node_t* node = base_sprite_inherit_t::sprite_list.get_node_by_reference(sprite_id_);
			//		auto sprite_node = &node->data;

			//		if (sprite_node->physics_nodereference != -1) {
			//			fan_2d::physics::sprite<user_struct_t>::erase(sprite_id_, sprite_node->physics_nodereference);
			//		}

			//		if (sprite_node->graphics_nodereference == -1) { // graphics or physics?
			//			base_sprite_inherit_t::sprite_list.unlink(sprite_id_);
			//			return;
			//		}

			//		fan_2d::opengl::sprite_t::erase(sprite_node->graphics_nodereference);
			//		uint32_t old_ref = sprite_node->graphics_nodereference;

			//		node = base_sprite_inherit_t::sprite_list.get_node_by_reference(base_sprite_inherit_t::sprite_list.begin());

			//		while (node->next != base_sprite_inherit_t::sprite_list.dst) {
			//			if (base_sprite_inherit_t::sprite_list[node->next].graphics_nodereference != (uint32_t)-1 && 
			//				base_sprite_inherit_t::sprite_list[node->next].graphics_nodereference > old_ref) {
			//				base_sprite_inherit_t::sprite_list[node->next].graphics_nodereference--;
			//			}
			//			node = base_sprite_inherit_t::sprite_list.get_node_by_reference(node->next);
			//		}

			//		base_sprite_inherit_t::sprite_list.unlink(sprite_id_);
			//	};

			//	if (inherit_t::m_engine->world.is_locked()) {
			//		inherit_t::m_engine->m_queue_after_step.push_back(f);
			//	}
			//	else {
			//		f();
			//	}

			//}

		};

		//struct circle : public base_engine<fan_2d::opengl::circle, fan_2d::physics::circle> {

		//	circle(fan_2d::engine::engine_t* engine) : circle::base_engine(engine) {
		//		engine->push_update(this, [&] { circle::update_position(); });
		//	}

		//	struct properties_t :
		//		public fan_2d::opengl::circle::properties_t {
		//		fan_2d::physics::body_type_t body_type;
		//		fan_2d::physics::body_property_t body_property = { 1, 10, 0.1 };
		//	};

		//	void push_back(const properties_t& properties) {

		//		fan_2d::opengl::circle::push_back(properties);

		//		fan_2d::physics::circle::properties_t c_properties;
		//		c_properties.position = properties.position / meter_scale;
		//		c_properties.radius = properties.radius;
		//		c_properties.body_property = properties.body_property;
		//		c_properties.body_type = properties.body_type;

		//		fan_2d::physics::circle::push_back(c_properties);
		//	}

		//	void set_position(uint32_t reference, const fan::vec2& position) {
		//		fan_2d::opengl::circle::set_position(i, position * meter_scale);

		//		fan_2d::physics::circle::set_position(i, position / meter_scale);
		//	}

		//	void update_position() {

		//		for (int i = 0; i < fan_2d::opengl::circle::size(); i++) {

		//			const fan::vec2 new_position = this->get_body(i)->get_position();

		//			if (this->get_body(i)->get_body_type() == fan_2d::physics::body_type_t::static_body ||
		//				fan_2d::opengl::circle::get_position(i) == new_position
		//				) {
		//				continue;
		//			}

		//			fan_2d::opengl::circle::set_position(i, new_position);
		//		}
		//	}
		//};

		//struct convex : public base_engine<fan_2d::opengl::convex, fan_2d::physics::convex> {

		//	convex(fan_2d::engine::engine_t* engine) : convex::base_engine(engine) {
		//		engine->push_update(this, [&] { convex::update_position(); });
		//	}

		//	struct properties_t : 
		//		public fan_2d::opengl::convex::properties_t
		//	{
		//		fan_2d::physics::body_type body_type;
		//		fan_2d::physics::body_property body_property = { 1, 10, 0.1 };
		//	};

		//	void push_back(const properties_t& properties) {

		//		fan_2d::opengl::convex::push_back(properties);

		//		fan_2d::physics::convex::properties_t c_properties;
		//		c_properties.position = properties.position;
		//		c_properties.angle = properties.angle;
		//		c_properties.points = (fan::vec2*)properties.points.data();
		//		c_properties.points_amount = properties.points.size();
		//		c_properties.body_property = properties.body_property;
		//		c_properties.body_type = properties.body_type;

		//		fan_2d::physics::convex::push_back(c_properties);
		//	}

		//	//void set_position(uint32_t reference, const fan::vec2& position) {

		//	//	const auto offset = position - fan_2d::opengl::vertice_vector::get_position(i);

		//	//	for (int j = 0; j < convex_amount; j++) {


		//	//		fan_2d::opengl::vertice_vector::set_position(i + j, fan_2d::opengl::vertice_vector::get_position(i + j) + offset);

		//	//		fan_2d::opengl::vertice_vector::set_angle(i + j, -physics::convex::get_body(i)->get_angle());

		//	//		fan::print(i, fan_2d::opengl::vertice_vector::get_position(i) + offset);

		//	//	}

		//	//	

		//	//	//fan_2d::physics::convex::set_position(i, position);
		//	//}

		//	//void update_position() {

		//	//	for (int i = 0; i < fan_2d::opengl::convex::size(); i++) {

		//	//		auto body = this->get_body(i / fan_2d::opengl::convex::convex_amount[i]);
		//	//	//	fan::print(body->get_position());
		//	//		fan_2d::opengl::convex::set_position(i, fan::vec2(body->get_position()));
		//	//		fan_2d::opengl::convex::set_angle(i, -body->get_angle());
		//	//	}

		//	//	fan_2d::opengl::convex::write_data();

		//	//}

		//};

		/*struct rope : public base_engine<fan_2d::opengl::line, fan_2d::physics::rope>{

		rope(engine_t* engine) : base_engine(engine) {

		}

		void push_back(std::vector<b2Vec2>& joints, const fan::color& color) {

		fan_2d::opengl::line::push_back(joints[0], joints[1], color);

		for (int i = 1; i < int(joints.size() / 2); i++) {
		fan_2d::opengl::line::push_back(joints[i], joints[i + 1], color);
		}

		fan_2d::physics::rope::push_back(joints);
		}

		fan::vec2 step(f32_t time_step) {
		b2Vec2 p(0, 0);
		m_rope.Step(time_step, 6, p);

		return p;
		}

		void update_position() {

		for (int i = 0; i < fan_2d::opengl::line::size(); i++) {
		fan_2d::opengl::line::set_line(
		i, 
		fan::vec2(fan_2d::physics::rope::rectangle::get_body(i)->GetPosition()),
		fan::vec2(fan_2d::physics::rope::rectangle::get_body(i + 1)->GetPosition())
		);
		}

		}

		};*/

	/*	struct motor_joint : public fan_2d::physics::motor_joint {
		public:

			motor_joint(fan_2d::engine::engine_t* engine) : fan_2d::physics::motor_joint(&engine->world) {

			}

			void push_back(fan_2d::body* a_body, fan_2d::body* b_body) {
				fan_2d::physics::motor_joint::push_back(a_body, b_body);
			}

			void erase(uintptr_t i) {
				fan_2d::physics::motor_joint::erase(i);
			}
			void erase(uintptr_t begin, uintptr_t end) {
				fan_2d::physics::motor_joint::erase(begin * 2, end * 2);
			}

			auto size() const {
				return wheel_joints.size();
			}

		};*/

}