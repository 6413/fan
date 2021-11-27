#pragma once

#include <fan/graphics/graphics.hpp>
#include <fan/physics/physics.hpp>

#include <fan/bll.hpp>

namespace fan_2d {

	namespace engine {
		
		struct engine_t {

			fan::window window;
			fan::camera camera;
			fan_2d::world* world = nullptr;

			std::unordered_map<void*, bool> m_to_avoid;

			std::unordered_map<void*, std::function<void()>> m_to_update;

			std::vector<std::function<void()>> m_to_remove;

			engine_t(const fan::vec2& gravity) : camera(&window), world(new fan_2d::world(gravity.b2())) {}

			~engine_t() {
				if (world) {
					delete world;
					world = nullptr;
				}
			}

			void step(f32_t time_step) {

				for (int i = 0; i < m_to_remove.size(); i++) {
					if (m_to_remove[i]) {
						m_to_remove[i]();
					}
				}

				m_to_remove.clear();

				world->Step(time_step, 6, 2);

				for (auto& i : m_to_update) {
					auto found = m_to_avoid.find(i.first);
					if (found == m_to_avoid.end() && i.second) {
						i.second();
					}
				}

				m_to_avoid.clear();
			}

			void avoid_updating(void* shape) {
				m_to_avoid.insert(std::make_pair(shape, false));
			}

			void push_update(void* shape, std::function<void()> update_function) {
				m_to_update.insert(std::make_pair(shape, update_function));
			}

		};

		template <typename graphics_t, typename physics_t>
		struct base_engine : public graphics_t, public physics_t {

			base_engine(fan_2d::engine::engine_t* engine) : m_engine(engine), graphics_t(&engine->camera), physics_t(engine->world) { }

			void set_rotation(uintptr_t i, f_t angle) {

				graphics_t::set_angle(i, angle);

				physics_t::set_angle(i, angle);

			}

			void erase(uintptr_t i) {
				std::function<void()> f = [&, i_ = i] {
					graphics_t::erase(i_);
					physics_t::erase(i_);
				};

				m_engine->m_to_remove.emplace_back(f);
			}
			void erase(uintptr_t begin, uintptr_t end) {
				std::function<void()> f = [&, begin_ = begin, end_ = end] {
					graphics_t::erase(begin_, end_);
					physics_t::erase(begin_, end_);
				};

				m_engine->m_to_remove.emplace_back(f);
			}

			fan_2d::engine::engine_t* m_engine;

		};

		struct rectangle : public base_engine<fan_2d::graphics::rectangle, fan_2d::physics::rectangle> {

			rectangle(fan_2d::engine::engine_t* engine) : rectangle::base_engine(engine) {
				engine->push_update(this, [&] { rectangle::update_position(); });
			}

			struct properties_t : public fan_2d::graphics::rectangle::properties_t {
				fan_2d::physics::body_type body_type;
				fan_2d::physics::body_property body_property = { 10, 1, 0.1 };		
			};

			void push_back(properties_t properties) {
				
				properties.rotation_point = properties.size / 2;

				fan_2d::graphics::rectangle::push_back(properties);

				fan_2d::physics::rectangle::properties_t p_property;
				p_property.position = properties.position;
				p_property.size = properties.size;
				p_property.angle = properties.angle;
				p_property.body_property = properties.body_property;
				p_property.body_type = properties.body_type;

				fan_2d::physics::rectangle::push_back(p_property);
			}

			void set_position(uintptr_t i, const fan::vec2& position) {
				fan_2d::graphics::rectangle::set_position(i, position - get_size(i) * 0.5);
				fan_2d::physics::rectangle::set_position(i, position);
			}

			void update_position() {

				for (int i = 0; i < this->size(); i++) {
					const fan::vec2 new_position = this->get_body(i)->get_position() * meter_scale - fan_2d::graphics::rectangle::get_size(i) * 0.5;

					const f32_t new_angle = -this->get_body(i)->get_angle();

					if (this->get_body(i)->get_body_type() == fan_2d::physics::body_type::static_body ||
						fan_2d::graphics::rectangle::get_position(i) == new_position
						&& fan_2d::graphics::rectangle::get_angle(i) == new_angle) {
						continue;
					}

					fan_2d::graphics::rectangle::set_position(i, new_position);
					fan_2d::graphics::rectangle::set_angle(i, new_angle);
				}

				this->write_data();
			}

		};

		struct empty {

		};

		template <typename user_struct_t>
		struct sprite_node_t {
			uint32_t graphics_nodereference;
			uint32_t physics_nodereference;
			uint32_t fixture_nodereference;
			fan::vec2 offset;
			user_struct_t user_data;
		};

		template <typename user_struct_t = empty>
		struct base_sprite_t : public base_engine<fan_2d::graphics::sprite, fan_2d::physics::sprite> {

		private:

			using fan_2d::graphics::sprite::push_back;
			using fan_2d::physics::sprite::physics_size;

		protected:

			bll_t<sprite_node_t<user_struct_t>, uint32_t> sprite_list;

		public:

			struct properties_t : public fan_2d::graphics::sprite::properties_t {

				fan_2d::physics::body_type body_type;
				fan_2d::physics::body_property body_property = { 10, 1, 0.1 };		
			};

			base_sprite_t(fan_2d::engine::engine_t* engine) : base_sprite_t::base_engine(engine) {
				engine->push_update(this, [&] { base_sprite_t::update_position(); });
			}

			fan::vec2 get_position(uint32_t sprite_id) const {
				return fan_2d::graphics::sprite::get_position(sprite_list[sprite_id].graphics_nodereference);
			}

			fan::vec2 get_size(uint32_t sprite_id) const {
				return fan_2d::graphics::sprite::get_size(sprite_list[sprite_id].graphics_nodereference);
			}

			void apply_force(uint32_t sprite_id, const fan::vec2& force, const fan::vec2& point = 0) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->apply_force(force * meter_scale, point);
			}

			void apply_impulse(uint32_t sprite_id, const fan::vec2& force, const fan::vec2& point = 0) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->apply_impulse(force * meter_scale, point);
			}

			void apply_torque(uint32_t sprite_id, f32_t torque) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->apply_torque(torque * meter_scale);
			}

			f32_t get_angular_damping(uint32_t sprite_id) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_angular_damping() / meter_scale;
			}
			void set_angular_damping(uint32_t sprite_id, f32_t amount) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_angular_damping(amount * meter_scale);
			}

			f32_t get_angular_velocity(uint32_t sprite_id) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_angular_velocity() / meter_scale;
			}
			void set_angular_velocity(uint32_t sprite_id, f32_t velocity) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_angular_velocity(velocity * meter_scale);
			}

			fan_2d::fixture* get_fixture_list(uint32_t sprite_id) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return (fan_2d::fixture*)inherit_t::m_body[node->data.physics_nodereference]->get_fixture_list();
			}

			f32_t get_gravity_scale(uint32_t sprite_id) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_gravity_scale() / meter_scale;
			}
			void set_gravity_scale(uint32_t sprite_id, f32_t scale) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_gravity_scale(scale * meter_scale);
			}

			f32_t get_velocity_damping(uint32_t sprite_id) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_velocity_damping() / meter_scale;
			}
			void set_velocity_damping(uint32_t sprite_id, f32_t amount) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_velocity_damping(amount * meter_scale);
			}

			fan::vec2 get_velocity(uint32_t sprite_id) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_velocity() / meter_scale;
			}
			void set_velocity(uint32_t sprite_id, const fan::vec2& velocity) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_velocity(velocity * meter_scale);
			}

			fan_2d::physics::body_type get_body_type(uint32_t sprite_id) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return (fan_2d::physics::body_type)inherit_t::m_body[node->data.physics_nodereference]->get_body_type();
			}

			f32_t get_density(uint32_t sprite_id) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_fixture[node->data.physics_nodereference]->get_density();
			}
			void set_density(uint32_t sprite_id, f32_t density) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_fixture[node->data.physics_nodereference]->set_density(density);
			}

			f32_t get_mass(uint32_t sprite_id) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->get_mass();
			}

			bool is_bullet(uint32_t sprite_id) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->is_bullet();
			}

			bool is_awake(uint32_t sprite_id) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->is_awake();
			}
			void set_awake(uint32_t sprite_id, bool flag) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->set_awake(flag);
			}

			bool is_enabled(uint32_t sprite_id) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_body[node->data.physics_nodereference]->is_enabled();
			}
			void set_enabled(uint32_t sprite_id, bool flag) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_body[node->data.physics_nodereference]->set_enabled(flag);
			}

			f32_t get_restitution(uint32_t sprite_id) const {
				const typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				return inherit_t::m_fixture[node->data.physics_nodereference]->get_restitution() / meter_scale;
			}
			void set_restitution(uint32_t sprite_id, f32_t restitution) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				inherit_t::m_fixture[node->data.physics_nodereference]->set_restitution(restitution * meter_scale);
			}

			void set_position(uint32_t sprite_id, const fan::vec2& position) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				this->inherit_t::m_body[node->data.physics_nodereference]->set_transform((position / meter_scale).b2(), this->inherit_t::m_body[node->data.physics_nodereference]->get_angle());
				this->inherit_t::m_body[node->data.physics_nodereference]->set_awake(true);
			}

			void set_angle(uint32_t sprite_id, f64_t rotation) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				this->inherit_t::m_body[node->data.physics_nodereference]->set_transform(this->inherit_t::m_body[node->data.physics_nodereference]->get_position(), -rotation);
			}

			void set_angular_rotation(uint32_t sprite_id, f64_t w) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				this->inherit_t::m_body[node->data.physics_nodereference]->set_angular_velocity(w);
				this->inherit_t::m_body[node->data.physics_nodereference]->set_awake(true);
			}

			using sprite = base_sprite_t<>;


			/*void erase(uintptr_t i) {
				decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);
				m_world->DestroyBody((b2Body*)inherit_t::get_body(i));

				inherit_t::m_fixture.erase(inherit_t::m_fixture.begin() + i);
				inherit_t::m_body.erase(inherit_t::m_body.begin() + i);

			}*/
			/*void erase(uintptr_t begin, uintptr_t end) {
				decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_id);

				if (begin > inherit_t::m_body.size() || end > inherit_t::m_body.size()) {
					return;
				}

				for (int i = begin; i < end; i++) {
					m_world->DestroyBody((b2Body*)inherit_t::get_body(i));
				}

				inherit_t::m_fixture.erase(inherit_t::m_fixture.begin() + begin, inherit_t::m_fixture.begin() + end);
				inherit_t::m_body.erase(inherit_t::m_body.begin() + begin, inherit_t::m_body.begin() + end);
			}*/

			/*void push_back(fan_2d::graphics::sprite::properties_t properties) {

			}*/

			void update_position() {

				for (int i = sprite_list.get_node_first(); i != sprite_list.dst; ) {
					
					// if graphics
					if (sprite_list[i].graphics_nodereference == (uint32_t)-1) {
						typename decltype(sprite_list)::node_t *node = sprite_list.get_node_by_reference(i);
						i = node->next;
						continue;
					}

					
					
					fan::vec2 new_position = this->m_body[sprite_list[i].physics_nodereference]->get_position() * meter_scale;

					if (this->m_body[sprite_list[i].physics_nodereference]->get_shape_type() == fan_2d::shape_type::rectangle) {
						//fan::print(aa_bb[1].x);

						new_position -= sprite_list[i].offset;
					}

					f32_t angle = -this->m_body[sprite_list[i].physics_nodereference]->get_angle();

					/*if (this->get_body(i)->get_body_type() == fan_2d::physics::body_type::static_body ||
						fan_2d::graphics::sprite::get_position(i) == new_position ||
						fan_2d::graphics::sprite::get_angle(i) == angle
						) {
						continue;
					}
					fan::print("update");*/
					fan_2d::graphics::sprite::set_angle(sprite_list[i].graphics_nodereference, angle);
					fan_2d::graphics::sprite::set_position(sprite_list[i].graphics_nodereference, new_position);

					typename decltype(sprite_list)::node_t *node = sprite_list.get_node_by_reference(i);
					i = node->next;
				} 

				this->write_data();

			}

			void sprite_init_graphics(uint32_t nodereference, const fan_2d::graphics::sprite::properties_t& properties) {

				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(nodereference);
				sprite_node_t<user_struct_t>* sprite_node = &node->data;
				if (sprite_node->graphics_nodereference != -1) {
					// already exists
					assert(0);
				}

				sprite_node->graphics_nodereference = fan_2d::graphics::sprite::size();

				fan_2d::graphics::sprite::push_back(properties);
			}
			void physics_add_triangle(uint32_t SpriteID, const fan_2d::physics::triangle::properties_t& properties) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(SpriteID);
				sprite_node_t<user_struct_t>* sprite_node = &node->data;
				if (sprite_node->physics_nodereference == (uint32_t)-1) {
					sprite_node->physics_nodereference = m_body.new_node_last();
					m_body[sprite_node->physics_nodereference] = nullptr;
				}
				sprite_node->fixture_nodereference = fan_2d::physics::sprite::physics_add_triangle(sprite_node->physics_nodereference, SpriteID, properties);
			}
			void physics_add_rectangle(uint32_t SpriteID, const fan_2d::physics::rectangle::properties_t& properties) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(SpriteID);
				sprite_node_t<user_struct_t>* sprite_node = &node->data;
				if (sprite_node->physics_nodereference == (uint32_t)-1) {
					sprite_node->physics_nodereference = m_body.new_node_last();
					m_body[sprite_node->physics_nodereference] = nullptr;
				}
				//sprite_node->physics_nodereference = physics_size();
				if (sprite_node->graphics_nodereference != (uint32_t)-1) {
				sprite_node->fixture_nodereference = fan_2d::physics::sprite::physics_add_rectangle(fan_2d::graphics::sprite::get_position(sprite_node->graphics_nodereference) + fan_2d::graphics::sprite::get_size(sprite_node->graphics_nodereference) / 2, sprite_node->physics_nodereference, SpriteID, properties);
					sprite_node->offset = fan_2d::graphics::sprite::get_size(sprite_node->graphics_nodereference) / 2;
				}
				else {
					sprite_node->fixture_nodereference = fan_2d::physics::sprite::physics_add_rectangle(properties.position + properties.size / 2, sprite_node->physics_nodereference, SpriteID, properties);
				}
			}
			void sprite_init_physics(uint32_t nodereference) {
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(nodereference);
				sprite_node_t<user_struct_t>* sprite_node = &node->data;
				if (sprite_node->physics_nodereference != (uint32_t)-1) {
					// already exists
					assert(0);
				}
				sprite_node->physics_nodereference = physics_size();

				// push_back things here
			}
			//void sprite_init_physics(uint32_t nodereference) {
			//	decltype(bll)::node_t* node = bll.get_node_by_reference(nodereference);
			//	sprite_node_t<user_struct_t>* sprite_node = &node->data;
			//	if (sprite_node->physics_nodereference != -1) {
			//		// already exists
			//		assert(0);
			//	}
			//	sprite_node->physics_nodereference = physics_size();

			//	// push_back things here
			//}

			const user_struct_t get_user_data(uint32_t sprite_id) const {
				return sprite_list[sprite_id].user_data;
			}

			user_struct_t& get_user_data(uint32_t sprite_id) {
				return sprite_list[sprite_id].user_data;
			}

			uint32_t push_sprite(user_struct_t user_struct) {
				uint32_t nodereference = sprite_list.new_node_last();
				typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(nodereference);
				sprite_node_t<user_struct_t>* sprite_node = &node->data;
				sprite_node->user_data = user_struct;
				sprite_node->graphics_nodereference = -1;
				sprite_node->physics_nodereference = -1;

				return nodereference;
			}
			void erase(uint32_t sprite_reference) {

				auto f = [&, sprite_reference = sprite_reference] {

					typename decltype(sprite_list)::node_t* node = sprite_list.get_node_by_reference(sprite_reference);
					sprite_node_t<user_struct_t>* sprite_node = &node->data;

					if (sprite_node->graphics_nodereference != -1) {
						fan_2d::graphics::sprite::erase(sprite_node->graphics_nodereference);
					}
					if (sprite_node->physics_nodereference != -1) {
						fan_2d::physics::sprite::erase(sprite_node->physics_nodereference);
					}

					while (node->next != sprite_list.dst) {
						sprite_list[node->next].graphics_nodereference--;
						node = sprite_list.get_node_by_reference(node->next);
					}

					sprite_list.unlink(sprite_reference);
				};

				m_engine->m_to_remove.push_back(f);

			}

		};

		struct circle : public base_engine<fan_2d::graphics::circle, fan_2d::physics::circle> {

			circle(fan_2d::engine::engine_t* engine) : circle::base_engine(engine) {
				engine->push_update(this, [&] { circle::update_position(); });
			}

			struct properties_t :
				public fan_2d::graphics::circle::properties_t {
				fan_2d::physics::body_type body_type;
				fan_2d::physics::body_property body_property = { 1, 10, 0.1 };
			};

			void push_back(const properties_t& properties) {
				
				fan_2d::graphics::circle::push_back(properties);

				fan_2d::physics::circle::properties_t c_properties;
				c_properties.position = properties.position;
				c_properties.radius = properties.radius;
				c_properties.body_property = properties.body_property;
				c_properties.body_type = properties.body_type;

				fan_2d::physics::circle::push_back(c_properties);
			}

			void set_position(uint32_t i, const fan::vec2& position) {
				fan_2d::graphics::circle::set_position(i, position);

				fan_2d::physics::circle::set_position(i, position);
			}

			void update_position() {

				for (int i = 0; i < fan_2d::graphics::circle::size(); i++) {

					const fan::vec2 new_position = this->get_body(i)->get_position() * meter_scale;

					if (this->get_body(i)->get_body_type() == fan_2d::physics::body_type::static_body ||
						fan_2d::graphics::circle::get_position(i) == new_position
						) {
						continue;
					}

					fan_2d::graphics::circle::set_position(i, new_position);
				}

				this->write_data();

			}
		};

		//struct convex : public base_engine<fan_2d::graphics::convex, fan_2d::physics::convex> {

		//	convex(fan_2d::engine::engine_t* engine) : convex::base_engine(engine) {
		//		engine->push_update(this, [&] { convex::update_position(); });
		//	}

		//	struct properties_t : 
		//		public fan_2d::graphics::convex::properties_t
		//	{
		//		fan_2d::physics::body_type body_type;
		//		fan_2d::physics::body_property body_property = { 1, 10, 0.1 };
		//	};

		//	void push_back(const properties_t& properties) {

		//		fan_2d::graphics::convex::push_back(properties);

		//		fan_2d::physics::convex::properties_t c_properties;
		//		c_properties.position = properties.position;
		//		c_properties.angle = properties.angle;
		//		c_properties.points = (fan::vec2*)properties.points.data();
		//		c_properties.points_amount = properties.points.size();
		//		c_properties.body_property = properties.body_property;
		//		c_properties.body_type = properties.body_type;

		//		fan_2d::physics::convex::push_back(c_properties);
		//	}

		//	//void set_position(uint32_t i, const fan::vec2& position) {

		//	//	const auto offset = position - fan_2d::graphics::vertice_vector::get_position(i);

		//	//	for (int j = 0; j < convex_amount; j++) {


		//	//		fan_2d::graphics::vertice_vector::set_position(i + j, fan_2d::graphics::vertice_vector::get_position(i + j) + offset);

		//	//		fan_2d::graphics::vertice_vector::set_angle(i + j, -physics::convex::get_body(i)->get_angle());

		//	//		fan::print(i, fan_2d::graphics::vertice_vector::get_position(i) + offset);

		//	//	}

		//	//	

		//	//	//fan_2d::physics::convex::set_position(i, position / meter_scale);
		//	//}

		//	//void update_position() {

		//	//	for (int i = 0; i < fan_2d::graphics::convex::size(); i++) {

		//	//		auto body = this->get_body(i / fan_2d::graphics::convex::convex_amount[i]);
		//	//	//	fan::print(body->get_position());
		//	//		fan_2d::graphics::convex::set_position(i, fan::vec2(body->get_position()) * meter_scale);
		//	//		fan_2d::graphics::convex::set_angle(i, -body->get_angle());
		//	//	}

		//	//	fan_2d::graphics::convex::write_data();

		//	//}

		//};

		/*struct rope : public base_engine<fan_2d::graphics::line, fan_2d::physics::rope>{

			rope(engine_t* engine) : base_engine(engine) {

			}

			void push_back(std::vector<b2Vec2>& joints, const fan::color& color) {

				fan_2d::graphics::line::push_back(joints[0], joints[1], color);

				for (int i = 1; i < int(joints.size() / 2); i++) {
					fan_2d::graphics::line::push_back(joints[i], joints[i + 1], color);
				}

				fan_2d::physics::rope::push_back(joints);
			}

			fan::vec2 step(f32_t time_step) {
				b2Vec2 p(0, 0);
				m_rope.Step(time_step, 6, p);

				return p;
			}

			void update_position() {

				for (int i = 0; i < fan_2d::graphics::line::size(); i++) {
					fan_2d::graphics::line::set_line(
						i, 
						fan::vec2(fan_2d::physics::rope::rectangle::get_body(i)->GetPosition()) * meter_scale,
						fan::vec2(fan_2d::physics::rope::rectangle::get_body(i + 1)->GetPosition()) * meter_scale
					);
				}

			}

		};*/

		struct motor_joint : public fan_2d::physics::motor_joint {
		public:

			motor_joint(fan_2d::engine::engine_t* engine) : fan_2d::physics::motor_joint(engine->world) {

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

		};

	}

}