#pragma once

//#include <fan/time/time.h>
//#include <fan/physics/collision/circle.h>
//
//namespace fan_2d {
//
//	namespace graphics {
//
//		static void draw_mode(fan::window_t* window, fill_mode_e fill_mode, face_e face) {
//
//#if fan_renderer == fan_renderer_opengl
//
//			glPolygonMode((GLenum)face, (GLenum)fill_mode);
//
//			polygon_fill_mode = fill_mode;
//			polygon_face = face;
//
//#elif fan_renderer == fan_renderer_vulkan
//
//			for (int i = 0; i < window->m_vulkan->pipelines.size(); i++) {
//				window->m_vulkan->pipelines[i]->flags.face = face;
//				window->m_vulkan->pipelines[i]->flags.fill_mode = fill_mode;
//				window->m_vulkan->pipelines[i]->recreate_pipeline(i, window->get_size(), window->m_vulkan->swapChainExtent);
//			}
//
//#endif
//
//		}
//
//		static fill_mode_e get_fill_mode(fan::window_t* window) {
//
//#if fan_renderer == fan_renderer_opengl
//
//			return polygon_fill_mode;
//
//#elif fan_renderer == fan_renderer_vulkan
//
//			return window->m_vulkan->pipelines[0].flags.fill_mode;
//
//#endif
//		}
//
//		static fan_2d::opengl::face_e get_face(fan::window_t* window) {
//
//#if fan_renderer == fan_renderer_opengl
//
//			return polygon_face;
//
//#elif fan_renderer == fan_renderer_vulkan
//
//			return window->m_vulkan->pipelines[0].flags.face;
//
//#endif
//		}
//
//		class rounded_rectangle : public fan_2d::opengl::vertice_vector {
//		public:
//
//			struct properties_t : public vertice_vector::properties_t{
//
//				fan::vec2 size;
//				f32_t radius = 0;
//			};
//
//			static constexpr int m_segments = 4 * 10;
//
//			rounded_rectangle(fan::camera* camera);
//
//			void push_back(const properties_t& properties);
//
//			fan::vec2 get_position(uintptr_t i) const;
//			void set_position(uintptr_t i, const fan::vec2& position);
//
//			fan::vec2 get_size(uintptr_t i) const;
//			void set_size(uintptr_t i, const fan::vec2& size);
//
//			f32_t get_radius(uintptr_t i) const;
//			void set_radius(uintptr_t i, f32_t radius);
//
//			bool inside(uintptr_t i) const;
//
//			fan::color get_color(uintptr_t i) const;
//			void set_color(uintptr_t i, const fan::color& color);
//
//			uint32_t size() const;
//
//			void write_data();
//
//			void edit_data(uint32_t i);
//
//			void edit_data(uint32_t begin, uint32_t end);
//
//			void enable_draw();
//			void disable_draw();
//
//		private:
//
//			using fan_2d::opengl::vertice_vector::push_back;
//
//			uint32_t total_points = 0;
//
//			std::vector<fan::vec2> m_position;
//			std::vector<fan::vec2> m_size;
//			std::vector<f32_t> m_radius;
//		};
//
//		class circle : public fan_2d::opengl::vertice_vector {
//		public:
//
//			circle(fan::camera* camera) : fan_2d::opengl::vertice_vector(camera) {}
//			~circle() {
//				if (m_draw_index != -1) {
//					m_camera->m_window->erase_draw_call(m_draw_index);
//					m_draw_index = -1;
//				}
//			}
//
//			struct properties_t {
//				fan::vec2 position;
//				f32_t radius; 
//				fan::color color;
//			};
//
//			void push_back(const properties_t& property) {
//				this->m_position.emplace_back(property.position);
//				this->m_radius.emplace_back(property.radius);
//
//				vertice_vector::properties_t properties;
//				properties.color = property.color;
//				properties.rotation_point = property.position + property.radius;
//
//				for (int i = 0; i < m_segments; i++) {
//
//					f32_t theta = fan::math::two_pi * f32_t(i) / m_segments;
//
//					properties.position = property.position + fan::vec2(property.radius * std::cos(theta), property.radius * std::sin(theta));
//
//					vertice_vector::push_back(properties);
//				}
//			}
//
//			fan::vec2 get_position(uintptr_t i) const {
//				return this->m_position[i];
//			}
//			void set_position(uintptr_t i, const fan::vec2& position) {
//				this->m_position[i] = position;
//
//				for (int j = 0; j < m_segments; j++) {
//
//					f32_t theta = fan::math::two_pi * f32_t(j) / m_segments;
//
//					vertice_vector::set_position(i * m_segments + j, position + fan::vec2(m_radius[i] * std::cos(theta), m_radius[i] * std::sin(theta)));
//				}
//			}
//
//			f32_t get_radius(uintptr_t i) const {
//				return this->m_radius[i];
//			}
//
//			void set_radius(uintptr_t i, f32_t radius) {
//				this->m_radius[i] = radius;
//
//				const fan::vec2 position = this->get_position(i);
//
//				for (int j = 0; j < m_segments; j++) {
//
//					f32_t theta = fan::math::two_pi * f32_t(j) / m_segments;
//
//					vertice_vector::set_position(i * m_segments + j, position + fan::vec2(m_radius[i] * std::cos(theta), m_radius[i] * std::sin(theta)));
//
//				}
//			}
//
//			bool inside(uintptr_t i, fan::vec2 position = fan::math::inf) const {
//
//				if (position == fan::math::inf) {
//					position = m_camera->m_window->get_mouse_position() + fan::vec2(m_camera->get_position());
//				}
//
//				return fan_2d::collision::circle::point_inside(position, this->get_position(i), m_radius[i]);
//			}
//
//			fan::color get_color(uintptr_t i) const {
//				return vertice_vector::get_color(i * m_segments);
//			}
//			void set_color(uintptr_t i, const fan::color& color) {
//				for (int j = 0; j < m_segments; j++) {
//					vertice_vector::set_color(i * m_segments + j, color);
//				}
//			}
//
//			uint32_t size() const {
//				return this->m_position.size();
//			}
//
//			void erase(uintptr_t i)  {
//				fan_2d::opengl::vertice_vector::erase(i * m_segments, i * m_segments + m_segments);
//
//				this->m_position.erase(this->m_position.begin() + i);
//				this->m_radius.erase(this->m_radius.begin() + i);
//			}
//			void erase(uintptr_t begin, uintptr_t end) {
//				fan_2d::opengl::vertice_vector::erase(begin * m_segments, end * m_segments);
//
//				this->m_position.erase(this->m_position.begin() + begin, this->m_position.begin() + end);
//				this->m_radius.erase(this->m_radius.begin() + begin, this->m_radius.begin() + end);
//			}
//			void clear() {
//				fan_2d::opengl::vertice_vector::clear();
//
//				this->m_position.clear();
//				this->m_radius.clear();
//			}
//
//			void set_draw_mode(fan_2d::opengl::fill_mode_e fill_mode)
//			{
//				m_fill_mode = fill_mode;
//			}
//
//			fan_2d::opengl::fill_mode_e get_draw_mode() const
//			{
//				return m_fill_mode;
//			}
//
//			void enable_draw() {
//				if (m_draw_index == -1 || m_camera->m_window->m_draw_queue[m_draw_index].first != this) {
//					m_draw_index = m_camera->m_window->push_draw_call(this, [&] {
//						this->draw();
//					});
//				}
//				else {
//					m_camera->m_window->edit_draw_call(m_draw_index, this, [&] {
//						this->draw();
//					});
//				}
//			}
//
//		protected:
//
//			void draw() {
//
//				if (m_fill_mode == fan_2d::opengl::get_fill_mode(m_camera->m_window)) {
//					fan_2d::opengl::vertice_vector::draw(fan_2d::opengl::shape::triangle_fan, m_segments, 0, this->size() * m_segments);
//
//					return;
//				}
//
//				auto fill_mode = fan_2d::opengl::get_fill_mode(m_camera->m_window);
//
//				fan_2d::opengl::draw_mode(m_camera->m_window, m_fill_mode, fan_2d::opengl::get_face(m_camera->m_window));
//
//				fan_2d::opengl::vertice_vector::draw(fan_2d::opengl::shape::triangle_fan, m_segments, 0, this->size() * m_segments);
//
//				fan_2d::opengl::draw_mode(m_camera->m_window, fill_mode, fan_2d::opengl::get_face(m_camera->m_window));
//			}
//
//			void write_data() {
//				vertice_vector::write_data();
//			}
//
//			void edit_data(uint32_t i) {
//				vertice_vector::edit_data(i * m_segments, i * m_segments + m_segments);
//			}
//
//			void edit_data(uint32_t begin, uint32_t end) {
//				vertice_vector::edit_data(begin * m_segments, (end - begin + 1) * m_segments);
//			}
//
//			static constexpr int m_segments = 50;
//
//			std::vector<fan::vec2> m_position;
//			std::vector<f32_t> m_radius;
//
//		};
//
//		struct particles : protected fan_2d::opengl::sprite_t {
//
//			struct particle_t {
//				fan::vec2 position;
//				fan::vec2 size;
//				fan::vec2 velocity;
//				f32_t angle;
//				f32_t angular_velocity;
//				image_t image;
//				// nanoseconds
//				uint64_t erase_time;
//				// 0-1 if 1, will fade untill particle erased, 0.5 half of that etc...
//				f32_t fade_point;
//			};
//
//			using fan_2d::opengl::sprite_t::sprite;
//
//			void push_back(const particle_t& particle) {
//				fan_2d::opengl::sprite_t::properties_t p;
//				p.image = particle.image;
//				p.position = particle.position;
//				p.size = particle.size;
//				p.angle = particle.angle;
//
//				fan_2d::opengl::sprite_t::push_back(p);
//
//				physics_t physics;
//				physics.velocity = particle.velocity;
//				physics.angular_velocity = particle.angular_velocity;
//				physics.erase_timer = fan::time::nanoseconds(particle.erase_time);
//				physics.erase_timer.start();
//				physics.fade_point = particle.fade_point;
//
//				m_physics.emplace_back(physics);
//			}
//
//			// returns if object was erased. if so call write_data else edit_data
//			bool update(uint32_t i, f32_t delta) {
//
//				bool edit_size = false;
//
//				if (m_physics[i].velocity != 0) {
//					fan_2d::opengl::sprite_t::set_position(i, fan_2d::opengl::sprite_t::get_position(i) + m_physics[i].velocity * delta);
//				}
//				if (m_physics[i].angular_velocity != 0) {
//					fan_2d::opengl::sprite_t::set_angle(i, fan_2d::opengl::sprite_t::get_angle(i) + m_physics[i].angular_velocity * delta);
//				}
//				if (m_physics[i].fade_point != 0) {
//					fan_2d::opengl::sprite_t::set_transparency(i, 1.0 - ((f32_t)m_physics[i].erase_timer.elapsed() / m_physics[i].erase_timer.m_time) * m_physics[i].fade_point);
//				}
//				if (m_physics[i].erase_timer.finished()) {
//					fan_2d::opengl::sprite_t::erase(i);
//					m_physics.erase(m_physics.begin() + i);
//					edit_size = true;
//				}
//
//				return edit_size;
//			}
//
//			using sprite::draw;
//			using sprite::size;
//
//		protected:
//
//			struct physics_t {
//				fan::vec2 velocity;
//				f32_t angular_velocity;
//
//				fan::time::clock erase_timer;
//				f32_t fade_point;
//			};
//
//			std::deque<physics_t> m_physics;
//
//		};
//
//	}
//
//}