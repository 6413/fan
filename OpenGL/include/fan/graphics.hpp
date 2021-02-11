#pragma once
//#ifndef __INTELLISENSE__ 


#define GLEW_STATIC
#include <GL/glew.h>

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

//#define FAN_PERFORMANCE
#define RAM_SAVER

#include <fan/types/da.hpp>
#include <fan/math/matrix.hpp>
#include <fan/color.hpp>
#include <fan/window/window.hpp>
#include <fan/math/math.hpp>
#include <fan/shader.h>
#include <fan/time.hpp>
#include <fan/network.hpp>
#include <fan/soil2/SOIL2.h>

namespace fan {

	#if SYSTEM_BIT == 32
	constexpr auto GL_FLOAT_T = GL_FLOAT;
	#else
	// for now
	constexpr auto GL_FLOAT_T = GL_FLOAT;
	#endif

	class camera {
	public:
		camera(fan::window* window);

		camera(const fan::camera& camera);
		camera(fan::camera&& camera);

		fan::camera& operator=(const fan::camera& camera);
		fan::camera& operator=(fan::camera&& camera) noexcept;

		void move(f_t movement_speed, bool noclip = true, f_t friction = 12);
		void rotate_camera(bool when);

		fan::mat4 get_view_matrix() const;
		fan::mat4 get_view_matrix(fan::mat4 m) const;

		fan::vec3 get_position() const;
		void set_position(const fan::vec3& position);

		fan::vec3 get_velocity() const;
		void set_velocity(const fan::vec3& velocity);

		f_t get_yaw() const;
		void set_yaw(f_t angle);

		f_t get_pitch() const;
		void set_pitch(f_t angle);

		bool first_movement = true;

		void update_view();

		static constexpr f_t sensitivity = 0.1;

		static constexpr f_t max_yaw = 180;
		static constexpr f_t max_pitch = 89;

		static constexpr f_t gravity = 500;
		static constexpr f_t jump_force = 100;

		static constexpr fan::vec3 world_up = fan::vec3(0, 1, 0);

		fan::window* m_window;

	private:

		fan::vec3 m_position;
		fan::vec3 m_front;

		f32_t m_yaw;
		f32_t m_pitch;
		fan::vec3 m_right;
		fan::vec3 m_up;
		fan::vec3 m_velocity;

	};

	template <typename T = fan::vec2>
	static T random_vector(f_t min, f_t max) {
		if constexpr (std::is_same_v<T, fan::vec2>) {
			return T(fan::random<int64_t, int64_t>(min, max), fan::random<int64_t, int64_t>(min, max));
		}
		else {
			return T(fan::random<int64_t, int64_t>(min, max), fan::random<int64_t, int64_t>(min, max), fan::random<int64_t, int64_t>(min, max));
		}
	}

	void bind_vao(uint32_t vao, const std::function<void()>& function);

	void write_glbuffer(unsigned int buffer, void* data, uint_t size, uint_t target = GL_ARRAY_BUFFER, uint_t location = fan::uninitialized);
	void edit_glbuffer(unsigned int buffer, void* data, uint_t offset, uint_t size, uint_t target = GL_ARRAY_BUFFER,  uint_t location = fan::uninitialized);

	enum class opengl_buffer_type {
		buffer_object,
		vertex_array_object,
		shader_storage_buffer_object,
		texture,
		last
	};

	template <opengl_buffer_type buffer_type, opengl_buffer_type I = opengl_buffer_type::last, typename ...T>
	constexpr auto comparer(const T&... x) {
		if constexpr (buffer_type == I) {
			std::get<static_cast<int>(I)>(std::forward_as_tuple(x...))();
		}
		else if (static_cast<int>(I) > 0) {
			comparer<buffer_type, static_cast<opengl_buffer_type>(static_cast<int>(I) - 1)>(x...);
		}
	}

	static void draw_2d(const std::function<void()>& function_) {
		glDisable(GL_DEPTH_TEST);
		function_();
		glEnable(GL_DEPTH_TEST);
	}

	#define private_template
	#define private_class_name vao_handler
	#define private_variable_name m_vao
	#define private_buffer_type opengl_buffer_type::vertex_array_object
	#define private_layout_location 0
	#include <fan/code_builder.hpp>

	#define private_template
	#define private_class_name ebo_handler
	#define private_variable_name m_ebo
	#define private_buffer_type opengl_buffer_type::buffer_object
	#define private_layout_location 0
	#include <fan/code_builder.hpp>

	#define private_template
	#define private_class_name texture_handler
	#define private_variable_name m_texture
	#define private_buffer_type opengl_buffer_type::texture
	#define private_layout_location 0
	#include <fan/code_builder.hpp>

	#define private_template template <uint_t T_layout_location, opengl_buffer_type T_buffer_type>
	#define private_class_name glsl_location_handler
	#define private_variable_name m_buffer_object
	#define private_buffer_type T_buffer_type
	#define private_layout_location T_layout_location
	#include <fan/code_builder.hpp>

	#define enable_function_for_vector 	   template<typename T = void, typename = typename std::enable_if<std::is_same<T, T>::value && enable_vector>::type>
	#define enable_function_for_non_vector template<typename T = void, typename = typename std::enable_if<std::is_same<T, T>::value && !enable_vector>::type>
	#define enable_function_for_vector_cpp template<typename T, typename enable_t>

	template <bool enable_vector, typename _Vector, uint_t layout_location = 1, opengl_buffer_type buffer_type = opengl_buffer_type::buffer_object>
	class basic_shape_position : public glsl_location_handler<layout_location, buffer_type> {
	public:

		basic_shape_position() : glsl_location_handler<layout_location, buffer_type>() {}

		basic_shape_position(const _Vector& position) : basic_shape_position()
		{
			if constexpr (enable_vector) {
				this->basic_push_back(position, true);
			}
			else {
				this->m_position = position;
			}
		}

		basic_shape_position(const basic_shape_position& vector) : glsl_location_handler<layout_location, buffer_type>(vector) {
			this->m_position = vector.m_position;
		}
		basic_shape_position(basic_shape_position&& vector) noexcept : glsl_location_handler<layout_location, buffer_type>(std::move(vector)) {
			this->m_position = std::move(vector.m_position);
		}

		basic_shape_position& operator=(const basic_shape_position& vector) {
			glsl_location_handler<layout_location, buffer_type>::operator=(vector);

			this->m_position = vector.m_position;

			return *this;
		}
		basic_shape_position& operator=(basic_shape_position&& vector) {
			glsl_location_handler<layout_location, buffer_type>::operator=(std::move(vector));

			this->m_position = std::move(vector.m_position);

			return *this;
		}

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void reserve(uint_t new_size) {
			m_position.reserve(new_size);
		}
		enable_function_for_vector void resize(uint_t new_size) {
			m_position.resize(new_size);
		}

		enable_function_for_vector std::vector<_Vector> get_positions() const {
			return this->m_position;
		}

		enable_function_for_vector void set_positions(const std::vector<_Vector>& positions) {
			this->m_position.clear();
			this->m_position.insert(this->m_position.begin(), positions.begin(), positions.end());
		}

		enable_function_for_vector _Vector get_position(uint_t i) const {
			return this->m_position[i];
		}
		enable_function_for_vector void set_position(uint_t i, const _Vector& position, bool queue = false) {
			this->m_position[i] = position;

			if (!queue) {
				this->edit_data(i);
			}
		}

		enable_function_for_vector void erase(uint_t i, bool queue = false) {
			this->m_position.erase(this->m_position.begin() + i);

			if (!queue) {
				this->write_data();
			}
		}
		enable_function_for_vector void erase(uint_t begin, uint_t end, bool queue = false) {
			if (!begin && end == this->m_position.size()) {
				this->m_position.clear();
			}
			else {
				this->m_position.erase(this->m_position.begin() + begin, this->m_position.begin() + end);
			}

			if (!queue) {
				this->write_data();
			}
		}

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector _Vector get_position() const {
			return m_position;
		}
		enable_function_for_non_vector void set_position(const _Vector& position) {
			this->m_position = position;
		}

		// -----------------------------------------------------

	protected:

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void initialize_buffers(bool divisor) {
			glsl_location_handler<layout_location, buffer_type>::initialize_buffers(m_position.data(), sizeof(_Vector) * m_position.size(), divisor, _Vector::size());
		}

		enable_function_for_vector void basic_push_back(const _Vector& position, bool queue = false) {
			this->m_position.emplace_back(position);
			if (!queue) {
				this->write_data();
			}
		}

		enable_function_for_vector void edit_data(uint_t i) {
			glsl_location_handler<layout_location, buffer_type>::edit_data(i, m_position.data() + i, sizeof(_Vector));
		}
		enable_function_for_vector void write_data() {
			glsl_location_handler<layout_location, buffer_type>::write_data(m_position.data(), sizeof(_Vector) * m_position.size());
		}

		// -----------------------------------------------------

		std::conditional_t<enable_vector, std::vector<_Vector>, _Vector> m_position;

	};
 
	template <bool enable_vector, typename _Vector, uint_t layout_location = 2, opengl_buffer_type buffer_type = opengl_buffer_type::buffer_object>
	class basic_shape_size : public glsl_location_handler<layout_location, buffer_type> {
	public:

		basic_shape_size()  : glsl_location_handler<layout_location, buffer_type>() {}
		basic_shape_size(const _Vector& size): fan::basic_shape_size<enable_vector, _Vector, layout_location, buffer_type>() {
			if constexpr (enable_vector) {
				this->basic_push_back(size, true);
			}
			else {
				this->m_size = size;
			}
		}

		basic_shape_size(const basic_shape_size& vector) : glsl_location_handler<layout_location, buffer_type>(vector) {
			this->m_size = vector.m_size;
		}
		basic_shape_size(basic_shape_size&& vector) noexcept : glsl_location_handler<layout_location, buffer_type>(std::move(vector)) {
			this->m_size = std::move(vector.m_size);
		}

		basic_shape_size& operator=(const basic_shape_size& vector) {
			glsl_location_handler<layout_location, buffer_type>::operator=(vector);

			this->m_size = vector.m_size;

			return *this;
		}
		basic_shape_size& operator=(basic_shape_size&& vector) noexcept {
			glsl_location_handler<layout_location, buffer_type>::operator=(std::move(vector));

			this->m_size = std::move(vector.m_size);

			return *this;
		}

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void reserve(uint_t new_size) {
			m_size.reserve(new_size);
		}
		enable_function_for_vector void resize(uint_t new_size) {
			m_size.resize(new_size);
		}

		enable_function_for_vector std::vector<_Vector> get_sizes() const {
			return m_size;
		}

		enable_function_for_vector _Vector get_size(uint_t i) const {
			return this->m_size[i];
		}
		enable_function_for_vector void set_size(uint_t i, const _Vector& size, bool queue = false) {	
			this->m_size[i] = size;

			if (!queue) {
				this->edit_data(i);
			}
		}

		enable_function_for_vector void erase(uint_t i, bool queue = false) {
			this->m_size.erase(this->m_size.begin() + i);

			if (!queue) {
				this->write_data();
			}
		}
		enable_function_for_vector void erase(uint_t begin, uint_t end, bool queue = false) {
			this->m_size.erase(this->m_size.begin() + begin, this->m_size.begin() + end);

			if (!queue) {
				this->write_data();
			}
		}

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector _Vector get_size() const {
			return m_size;
		}
		enable_function_for_non_vector void set_size(const _Vector& size) {
			this->m_size = size;
		}

		// -----------------------------------------------------

	protected:

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void basic_push_back(const _Vector& size, bool queue = false) {
			this->m_size.emplace_back(size);

			if (!queue) {
				this->write_data();
			}
		}

		enable_function_for_vector void edit_data(uint_t i) {
			glsl_location_handler<layout_location, buffer_type>::edit_data(i, m_size.data() + i, sizeof(_Vector));
		}

		enable_function_for_vector void write_data() {
			glsl_location_handler<layout_location, buffer_type>::write_data(m_size.data(), sizeof(_Vector) * m_size.size());
		}

		enable_function_for_vector void initialize_buffers(bool divisor) {
			glsl_location_handler<layout_location, buffer_type>::initialize_buffers(m_size.data(), vector_byte_size(m_size), divisor, _Vector::size());
		}

		// -----------------------------------------------------

		std::conditional_t<enable_vector, std::vector<_Vector>, _Vector> m_size;

	};

	template <bool enable_vector, uint_t layout_location = 0, opengl_buffer_type buffer_type = opengl_buffer_type::buffer_object>
	class basic_shape_color_vector : public glsl_location_handler<layout_location, buffer_type> {
	public:

		basic_shape_color_vector() : glsl_location_handler<layout_location, buffer_type>() {}
		basic_shape_color_vector(const fan::color& color) : basic_shape_color_vector() {
			if constexpr (enable_vector) {
				basic_push_back(color, true);
			}
			else {
				this->m_color = color;
			}
		}
		basic_shape_color_vector(const basic_shape_color_vector& vector) : glsl_location_handler<layout_location, buffer_type>(vector) {
			this->m_color = vector.m_color;
		}
		basic_shape_color_vector(basic_shape_color_vector&& vector) noexcept : glsl_location_handler<layout_location, buffer_type>(std::move(vector)) {
			this->m_color = std::move(vector.m_color);
		}

		basic_shape_color_vector& operator=(const basic_shape_color_vector& vector) {
			glsl_location_handler<layout_location, buffer_type>::operator=(vector);

			this->m_color = vector.m_color;

			return *this;
		}
		basic_shape_color_vector& operator=(basic_shape_color_vector&& vector) noexcept {
			glsl_location_handler<layout_location, buffer_type>::operator=(std::move(vector));

			this->m_color = std::move(vector.m_color);

			return *this;
		}

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void reserve(uint_t new_size) {
			m_color.reserve(new_size);
		}
		enable_function_for_vector void resize(uint_t new_size, const fan::color& color) {
			m_color.resize(new_size, color);
		}
		enable_function_for_vector void resize(uint_t new_size) {
			m_color.resize(new_size);
		}

		enable_function_for_vector fan::color get_color(uint_t i) {
			return this->m_color[i];
		}

		enable_function_for_vector void set_color(uint_t i, const fan::color& color, bool queue = false) {
			this->m_color[i] = color;
			if (!queue) {
				this->edit_data(i);
			}
		}

		enable_function_for_vector void erase(uint_t i, bool queue = false) {
			this->m_color.erase(this->m_color.begin() + i);

			if (!queue) {
				this->write_data();
			}
		}
		enable_function_for_vector void erase(uint_t begin, uint_t end, bool queue = false) {
			this->m_color.erase(this->m_color.begin() + begin, this->m_color.begin() + end);

			if (!queue) {
				this->write_data();
			}
		}

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector fan::color get_color() const {
			return this->m_color;
		}
		enable_function_for_non_vector void set_color(const fan::color& color) {
			this->m_color = color;
		}

		// -----------------------------------------------------

	protected:

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void basic_push_back(const fan::color& color, bool queue = false) {
			this->m_color.emplace_back(color);

			if (!queue) {
				this->write_data();
			}
		}
	
		enable_function_for_vector void edit_data(uint_t i) {
			glsl_location_handler<layout_location, buffer_type>::edit_data(i, m_color.data() + i, sizeof(fan::color));
		}
		enable_function_for_vector void edit_data(void* data, uint_t offset, uint_t byte_size) {
			basic_shape_color_vector::glsl_location_handler::edit_data(data, offset, byte_size);
		}

		enable_function_for_vector void write_data() {
			glsl_location_handler<layout_location, buffer_type>::write_data(m_color.data(), sizeof(fan::color) * m_color.size());
		}

		enable_function_for_vector void initialize_buffers(bool divisor) {
			glsl_location_handler<layout_location, buffer_type>::initialize_buffers(m_color.data(), sizeof(fan::color) * m_color.size(), divisor, fan::color::size());
		}

		// -----------------------------------------------------

		std::conditional_t<enable_vector, std::vector<fan::color>, fan::color> m_color;

	};

	template <bool enable_vector, typename _Vector>
	class basic_shape_velocity {
	public:

		basic_shape_velocity() : m_velocity(1) {}
		basic_shape_velocity(const _Vector& velocity) {
			if constexpr (enable_vector) {
				this->m_velocity.emplace_back(velocity);
			}
			else {
				this->m_velocity = velocity;
			}
		}

		basic_shape_velocity(const basic_shape_velocity& vector) {
			this->operator=(vector);
		}
		basic_shape_velocity(basic_shape_velocity&& vector) noexcept {
			this->operator=(std::move(vector));
		}

		basic_shape_velocity& operator=(const basic_shape_velocity& vector) {
			this->m_velocity = vector.m_velocity;

			return *this;
		}
		basic_shape_velocity& operator=(basic_shape_velocity&& vector) noexcept {
			this->m_velocity = std::move(vector.m_velocity);

			return *this;
		}

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector _Vector get_velocity(uint_t i) const {
			return this->m_velocity[i];
		}
		enable_function_for_vector void set_velocity(uint_t i, const _Vector& velocity) {
			this->m_velocity[i] = velocity;
		}

		enable_function_for_vector void reserve(uint_t new_size) {
			this->m_velocity.reserve(new_size);
		}
		enable_function_for_vector void resize(uint_t new_size) {
			this->m_velocity.resize(new_size);
		}

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector _Vector get_velocity() const {
			return this->m_velocity;
		}
		enable_function_for_non_vector void set_velocity(const _Vector& velocity) {
			this->m_velocity = velocity;
		}

		// -----------------------------------------------------

	protected:

		std::conditional_t<enable_vector, std::vector<_Vector>, _Vector> m_velocity;

	};

	template <bool enable_vector, typename _Vector>
	class basic_shape : 
		public basic_shape_position<enable_vector, _Vector>, 
		public basic_shape_size<enable_vector, _Vector>,
		public vao_handler {
	public:
		
		basic_shape(fan::camera* camera) : m_camera(camera) {}
		basic_shape(fan::camera* camera, const fan::shader& shader) : m_camera(camera), m_shader(shader) { }
		basic_shape(fan::camera* camera, const fan::shader& shader, const _Vector& position, const _Vector& size) 
			: basic_shape::basic_shape_position(position), basic_shape::basic_shape_size(size), m_camera(camera), m_shader(shader) { }

		basic_shape(const basic_shape& vector) : 
			basic_shape::basic_shape_position(vector),
	        basic_shape::basic_shape_size(vector), vao_handler(vector),
	        m_camera(vector.m_camera), m_shader(vector.m_shader) { }
		basic_shape(basic_shape&& vector) noexcept 
			: basic_shape::basic_shape_position(std::move(vector)), 
		      basic_shape::basic_shape_size(std::move(vector)), vao_handler(std::move(vector)),
		      m_camera(vector.m_camera), m_shader(std::move(vector.m_shader)) { }

		basic_shape& operator=(const basic_shape& vector) {
			basic_shape::basic_shape_position::operator=(vector);
			basic_shape::basic_shape_size::operator=(vector);
			vao_handler::operator=(vector);

			m_camera->operator=(*vector.m_camera);
			m_shader.operator=(vector.m_shader);

			return *this;
		}
		basic_shape& operator=(basic_shape&& vector) noexcept {
			basic_shape::basic_shape_position::operator=(std::move(vector));
			basic_shape::basic_shape_size::operator=(std::move(vector));
			vao_handler::operator=(std::move(vector));

			this->m_camera->operator=(std::move(*vector.m_camera));
			this->m_shader.operator=(std::move(vector.m_shader));

			return *this;
		}

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void reserve(uint_t new_size) {
			basic_shape::basic_shape_position::reserve(new_size);
			basic_shape::basic_shape_size::reserve(new_size);
		}
		enable_function_for_vector void resize(uint_t new_size) {
			basic_shape::basic_shape_position::resize(new_size);
			basic_shape::basic_shape_size::resize(new_size);
		}

		enable_function_for_vector uint_t size() const {
			return this->m_position.size();
		}

		// -----------------------------------------------------

		f_t get_delta_time() const {
			return m_camera->m_window->get_delta_time();
		}

		fan::camera* m_camera;

	protected:

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void basic_push_back(const _Vector& position, const _Vector& size, bool queue = false) {
			basic_shape::basic_shape_position::basic_push_back(position, queue);
			basic_shape::basic_shape_size::basic_push_back(size, queue);
		}

		enable_function_for_vector void erase(uint_t i, bool queue = false) {
			basic_shape::basic_shape_position::erase(i, queue);
			basic_shape::basic_shape_size::erase(i, queue);
		}
		enable_function_for_vector void erase(uint_t begin, uint_t end, bool queue = false) {
			basic_shape::basic_shape_position::erase(begin, end, queue);
			basic_shape::basic_shape_size::erase(begin, end, queue);
		}

		enable_function_for_vector void edit_data(uint_t i, bool position, bool size) {
			if (position) {
				basic_shape::basic_shape_position::edit_data(i);
			}
			if (size) {
				basic_shape::basic_shape_size::edit_data(i);
			}
		}

		enable_function_for_vector void write_data(bool position, bool size) {
			if (position) {
				basic_shape::basic_shape_position::write_data();
			}
			if (size) {
				basic_shape::basic_shape_size::write_data();
			}
		}

		enable_function_for_vector void basic_draw(unsigned int mode, uint_t count, uint_t primcount, uint_t i = fan::uninitialized) const {
			glBindVertexArray(m_vao);
			if (i != (uint_t)fan::uninitialized) {
				glDrawArraysInstancedBaseInstance(mode, 0, count, 1, i);
			}
			else {
				glDrawArraysInstanced(mode, 0, count, primcount);
			}

			glBindVertexArray(0);
		}

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector void basic_draw(unsigned int mode, uint_t count) const {
			glBindVertexArray(m_vao);
			glDrawArrays(mode, 0, count);
			glBindVertexArray(0);
		}

		// -----------------------------------------------------

		fan::shader m_shader;

	};

	template <typename _Vector>
	class basic_vertice_vector : 
		public basic_shape_position<true, _Vector>, 
		public basic_shape_color_vector<true>, 
		public basic_shape_velocity<true, _Vector>,
	    public vao_handler {
	public:

		basic_vertice_vector(fan::camera* camera, const fan::shader& shader);
		basic_vertice_vector(fan::camera* camera, const fan::shader& shader, const fan::vec2& position, const fan::color& color);

		basic_vertice_vector(const basic_vertice_vector& vector) ;
		basic_vertice_vector(basic_vertice_vector&& vector) noexcept;

		basic_vertice_vector& operator=(const basic_vertice_vector& vector);
		basic_vertice_vector& operator=(basic_vertice_vector&& vector);

		uint_t size() const;

		void erase(uint_t i, bool queue = false);
		void erase(uint_t begin, uint_t end, bool queue = false);

		fan::camera* m_camera;

	protected:

		void basic_push_back(const _Vector& position, const fan::color& color, bool queue = false);

		void edit_data(uint_t i, bool position, bool color);

		void write_data(bool position, bool color);

		void basic_draw(unsigned int mode, uint_t count) const;

		fan::shader m_shader;

	};

	template <uint_t layout_location = 0, uint_t gl_buffer = GL_ARRAY_BUFFER>
	class basic_shape_color_vector_vector : public glsl_location_handler<layout_location, opengl_buffer_type::buffer_object> {
	public:

		basic_shape_color_vector_vector();
		basic_shape_color_vector_vector(const std::vector<fan::color>& color);

		basic_shape_color_vector_vector(const basic_shape_color_vector_vector& vector_vector) 
			: basic_shape_color_vector_vector::glsl_location_handler(vector_vector), m_color(vector_vector.m_color) {}
		basic_shape_color_vector_vector(basic_shape_color_vector_vector&& vector_vector) 
			: basic_shape_color_vector_vector::glsl_location_handler(std::move(vector_vector)), m_color(std::move(vector_vector.m_color)) {}

		basic_shape_color_vector_vector& operator=(const basic_shape_color_vector_vector& vector_vector) {
			basic_shape_color_vector_vector::glsl_location_handler::operator=(vector_vector);
			m_color = vector_vector.m_color;

			return *this;
		}

		basic_shape_color_vector_vector& operator=(basic_shape_color_vector_vector&& vector_vector) {
			basic_shape_color_vector_vector::glsl_location_handler::operator=(std::move(vector_vector));
			m_color = std::move(vector_vector.m_color);

			return *this;
		}

		std::vector<fan::color> get_color(uint_t i);
		void set_color(uint_t i, const std::vector<fan::color>& color, bool queue = false);

		void erase(uint_t i, bool queue = false);


	protected:

		void basic_push_back(const std::vector<fan::color>& color, bool queue = false);
	
		void edit_data(uint_t i);
		void edit_data(void* data, uint_t offset, uint_t size);

		void write_data()
		{
			std::vector<fan::color> vector;

			for (uint_t i = 0; i < m_color.size(); i++) {
				for (int j = 0; j < 6; j++) {
					vector.insert(vector.end(), m_color[i].begin(), m_color[i].end());
				}
			}

			fan::write_glbuffer(glsl_location_handler<layout_location, opengl_buffer_type::buffer_object>::m_buffer_object, vector.data(), sizeof(fan::color) * vector.size(), gl_buffer, layout_location);
		}

		void initialize_buffers(bool divisor = true);

		std::vector<std::vector<fan::color>> m_color;

	};

	// editing this requires change in glsl file
	enum class e_shapes {
		VERTICE,
		LINE,
		SQUARE,
		TRIANGLE
	};

	using map_t = std::vector<std::vector<std::vector<bool>>>;

}

namespace fan_2d {

	fan::mat4 get_projection(const fan::vec2i& window_size);
	fan::mat4 get_view_translation(const fan::vec2i& window_size, const fan::mat4& view);

	namespace shader_paths {
		constexpr auto text_renderer_vs("glsl/2D/text.vs");
		constexpr auto text_renderer_fs("glsl/2D/text.fs");

		constexpr auto single_shapes_vs("glsl/2D/shapes.vs");
		constexpr auto single_shapes_fs("glsl/2D/shapes.fs");

		constexpr auto single_shapes_bloom_vs("glsl/2D/bloom.vs");
		constexpr auto single_shapes_bloom_fs("glsl/2D/bloom.fs");
		constexpr auto single_shapes_blur_vs("glsl/2D/blur.vs");
		constexpr auto single_shapes_blur_fs("glsl/2D/blur.fs");

		constexpr auto single_shapes_bloom_final_vs("glsl/2D/bloom_final.vs");
		constexpr auto single_shapes_bloom_final_fs("glsl/2D/bloom_final.fs");

		constexpr auto single_sprite_vs("glsl/2D/sprite.vs");
		constexpr auto single_sprite_fs("glsl/2D/sprite.fs");

		constexpr auto shape_vector_vs("glsl/2D/shape_vector.vs");
		constexpr auto shape_vector_fs("glsl/2D/shapes.fs");
		constexpr auto sprite_vector_vs("glsl/2D/sprite_vector.vs");
		constexpr auto sprite_vector_fs("glsl/2D/sprite_vector.fs");
	}

	// returns how much object moved
	fan::vec2 move_object(fan::window* window, fan::vec2& position, fan::vec2& velocity, f32_t speed, f32_t gravity, f32_t jump_force = -800, f32_t friction = 10);

	struct rectangle_corners_t {
		fan::vec2 top_left;
		fan::vec2 top_right;
		fan::vec2 bottom_left;
		fan::vec2 bottom_right;
		
		const fan::vec2 operator[](const uint_t i) const {
			return !i ? top_left : i == 1 ? top_right : i == 2 ? bottom_left : bottom_right;
		}

		fan::vec2& operator[](const uint_t i) {
			return !i ? top_left : i == 1 ? top_right : i == 2 ? bottom_left : bottom_right;
		}

	};

	// 0 left right, 1 top right, 2 bottom left, 3 bottom right

	constexpr rectangle_corners_t get_rectangle_corners(const fan::vec2& position, const fan::vec2& size) {
		return { position, position + fan::vec2(size.x, 0), position + fan::vec2(0, size.y), position + size };
	}

	struct line : 
		protected fan::basic_shape<0, fan::vec2>, 
		public fan::basic_shape_color_vector<0> {

		line(fan::camera* camera);
		line(fan::camera* camera, const fan::mat2& begin_end, const fan::color& color);

		void draw();

		fan::mat2 get_position() const;
		void set_position(const fan::mat2& begin_end);

	};

	struct rectangle : 
		public fan::basic_shape<0, fan::vec2>, 
		public fan::basic_shape_color_vector<0>, 
		public fan::basic_shape_velocity<0, fan::vec2> {

		rectangle(fan::camera* camera);
		rectangle(fan::camera* camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color);

		rectangle(const rectangle& single);
		rectangle(rectangle&& single) noexcept;

		rectangle& operator=(const rectangle& single);
		rectangle& operator=(rectangle&& single);

		rectangle_corners_t get_corners() const;

		fan::vec2 get_center() const;

		f_t get_rotation() const;
		void set_rotation(f_t angle);

		void draw() const;

		fan::vec2 move(f32_t speed, f32_t gravity = 0, f32_t jump_force = -800, f32_t friction = 10);

		//void set_position(const fan::vec2& position) override;

	private:

		rectangle_corners_t m_corners;

		f_t m_rotation;

	};

	struct image_info {
		fan::vec2i image_size;
		uint32_t texture_id;
	};

	namespace image_load_properties {
		inline uint_t internal_format = GL_RGBA;
		inline uint_t format = GL_RGBA;
		inline uint_t type = GL_UNSIGNED_BYTE;
	}

	static fan::vec2  load_image(uint32_t& texture_id, const std::string& path, bool flip_image = false);
	static image_info load_image(unsigned char* pixels, const fan::vec2i& size);

	class sprite : 
		public fan::basic_shape<0, fan::vec2>, 
		fan::basic_shape_velocity<0, fan::vec2>,
		fan::texture_handler {
	public:
		sprite(fan::camera* camera);

		// size with default is the size of the image
		sprite(fan::camera* camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0, f_t transparency = 1);
		sprite(fan::camera* camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size = 0, f_t transparency = 1);

		sprite(const fan_2d::sprite& sprite);
		sprite(fan_2d::sprite&& sprite) noexcept;

		fan_2d::sprite& operator=(const fan_2d::sprite& sprite);
		fan_2d::sprite& operator=(fan_2d::sprite&& sprite);

		void load_sprite(const std::string& path, const fan::vec2i& size = 0, bool flip_image = false);

		void reload_sprite(unsigned char* pixels, const fan::vec2i& size);
		void reload_sprite(const std::string& path, const fan::vec2i& size, bool flip_image = false);

		void draw();

		f32_t get_rotation();
		void set_rotation(f32_t degrees);

	private:

		f32_t m_rotation;
		f_t m_transparency;

		std::string m_path;
	};

	/*class animation : public basic_single_shape {
	public:

		animation(const fan::vec2& position, const fan::vec2& size);

		void add(const std::string& path);

		void draw(uint_t m_texture);

	private:
		std::vector<unsigned int> m_textures;
	};*/

	class vertice_vector : public fan::basic_vertice_vector<fan::vec2>, public fan::ebo_handler {
	public:

		vertice_vector(fan::camera* camera, uint_t index_restart = UINT32_MAX);
		vertice_vector(fan::camera* camera, const fan::vec2& position, const fan::color& color, uint_t index_restart);
		vertice_vector(const vertice_vector& vector);
		vertice_vector(vertice_vector&& vector) noexcept;

		vertice_vector& operator=(const vertice_vector& vector);
		vertice_vector& operator=(vertice_vector&& vector) noexcept;

		void release_queue(bool position, bool color, bool indices);

		virtual void push_back(const fan::vec2& position, const fan::color& color, bool queue = false);

		void draw(uint32_t mode) const;

		void erase(uint_t i, bool queue = false);
		void erase(uint_t begin, uint_t end, bool queue = false);

		void initialize_buffers();

	protected:

		void write_data();

		uint_t m_index_restart;

		std::vector<uint32_t> m_indices;

		uint_t m_offset;

	};

	class line_vector : 
		public fan::basic_shape<true, fan::vec2>, 
		public fan::basic_shape_color_vector<true> {
	public:
		
		line_vector(fan::camera* camera);
		line_vector(fan::camera* camera, const fan::mat2& begin_end, const fan::color& color);

		line_vector(const line_vector& vector);
		line_vector(line_vector&& vector) noexcept;

		line_vector& operator=(const line_vector& vector);
		line_vector& operator=(line_vector&& vector) noexcept;

		void reserve(uint_t size);
		void resize(uint_t size, const fan::color& color);

		void push_back(const fan::mat2& begin_end, const fan::color& color, bool queue = false);

		void draw(uint_t i = fan::uninitialized);

		void set_position(uint_t i, const fan::mat2& begin_end, bool queue = false);

		void release_queue(bool position, bool color);

		void initialize_buffers();

	private:
		using line_vector::basic_shape::set_position;
		using line_vector::basic_shape::set_size;
	};

	struct triangle_vector : public fan::basic_shape<true, fan::vec2>, public fan::basic_shape_color_vector<true> {

		triangle_vector();
		triangle_vector(const fan::mat3x2& corners, const fan::color& color);
		
		void set_position(uint_t i, const fan::mat3x2& corners);
		void push_back(const fan::mat3x2& corners, const fan::color& color);

		void draw();

	private:

	};

	class rectangle_vector : 
		public fan::basic_shape<true, fan::vec2>, 
		public fan::basic_shape_color_vector<true>, 
		public fan::basic_shape_velocity<true, fan::vec2> {
	public:

		rectangle_vector(fan::camera* camera);
		rectangle_vector(fan::camera* camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color);

		rectangle_vector(const rectangle_vector& vector);
		rectangle_vector(rectangle_vector&& vector) noexcept;

		rectangle_vector& operator=(const rectangle_vector& vector);
		rectangle_vector& operator=(rectangle_vector&& vector) noexcept;

		void initialize_buffers();

		fan_2d::rectangle construct(uint_t i);

		void release_queue(bool position, bool size, bool color);

		void push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, bool queue = false);
		void erase(uint_t i, bool queue = false);
		void erase(uint_t begin, uint_t end, bool queue = false);

		void draw(uint_t i = fan::uninitialized) const;

		fan::vec2 get_center(uint_t i) const;

		fan_2d::rectangle_corners_t get_corners(uint_t i) const;

		void move(uint_t i, f32_t speed, f32_t gravity = 0, f32_t jump_force = -800, f32_t friction = 10);

		bool inside(uint_t i) const;

		f_t get_rotation(uint_t i) const;
		void set_rotation(uint_t i, f_t angle);

	private:

		std::vector<fan_2d::rectangle_corners_t> m_corners;
		std::vector<f_t> m_rotation;

	};

	class sprite_vector : 
		public fan::basic_shape<true, fan::vec2>,
		public fan::basic_shape_velocity<true, fan::vec2>,
		public fan::texture_handler {
	public:

		sprite_vector(fan::camera* camera, const std::string& path);
		sprite_vector(fan::camera* camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0);
		sprite_vector(const sprite_vector& vector);
		sprite_vector(sprite_vector&& vector) noexcept;

		sprite_vector& operator=(const sprite_vector& vector);
		sprite_vector& operator=(sprite_vector&& vector) noexcept;

		void initialize_buffers();

		void push_back(const fan::vec2& position, const fan::vec2& size = 0, bool queue = false);

		void reserve(uint_t new_size);
		void resize(uint_t new_size);

		void draw();

		void release_queue(bool position, bool size);

		void erase(uint_t i, bool queue = false);

		void load_sprite(const std::string& path, const fan::vec2 size = 0);

	protected:

		void allocate_texture();

		// still on progress
		std::vector<f_t> m_rotation;

		fan::vec2i m_original_image_size;

		std::string m_path;

	};

	class rounded_rectangle : public fan_2d::vertice_vector {
	public:

		static constexpr f_t segments = 10;

		rounded_rectangle(fan::camera* camera);
		rounded_rectangle(fan::camera* camera, const fan::vec2& position, const fan::vec2& size, f_t radius, const fan::color& color);

		void push_back(const fan::vec2& position, const fan::vec2& size, f_t radius, const fan::color& color, bool queue = false);

		fan::vec2 get_position(uint_t i) const;
		void set_position(uint_t i, const fan::vec2& position, bool queue = false);

		fan::vec2 get_size(uint_t i) const;
		void set_size(uint_t i, const fan::vec2& size, bool queue = false);

		f_t get_radius(uint_t i) const;
		void set_radius(uint_t i, f_t radius, bool queue = false);

		void draw() const;

		bool inside(uint_t i) const;

		fan::color get_color(uint_t i) const;
		void set_color(uint_t i, const fan::color& color, bool queue = false);

		uint_t size() const;

	private:

		using fan_2d::vertice_vector::push_back;

		void edit_rectangle(uint_t i, bool queue = false);

		std::vector<fan::vec2> m_position;
		std::vector<fan::vec2> m_size;
		std::vector<f_t> m_radius;

		std::vector<uint_t> m_data_offset;

	};

	struct particle {
		fan::vec2 m_velocity;
		fan::timer<> m_timer; // milli
	};

	class particles : public fan_2d::rectangle_vector {
	public:

		particles(fan::camera* camera);

		void add(
			const fan::vec2& position, 
			const fan::vec2& size, 
			const fan::vec2& velocity, 
			const fan::color& color, 
			uint_t time
		);

		void update();

	private:

		std::vector<fan_2d::particle> m_particles;
	};

	fan_2d::line_vector create_grid(fan::camera* camera, const fan::vec2i& block_size, const fan::vec2i& grid_size, const fan::color& color);

	namespace gui {
		
		fan::vec2 get_resize_movement_offset(fan::window* window);

		void add_resize_callback(fan::window* window, fan::vec2& position);

		struct rectangle : public fan_2d::rectangle {

			rectangle(fan::camera* camera);
			rectangle(fan::camera* camera,const fan::vec2& position, const fan::vec2& size, const fan::color& color);

		};

		struct rectangle_vector : public fan_2d::rectangle_vector {

			rectangle_vector(fan::camera* camera);
			rectangle_vector(fan::camera* camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color);

		};

		struct sprite : public fan_2d::sprite {

			sprite(fan::camera* camera);
			// scale with default is sprite size
			sprite(fan::camera* camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0, f_t transparency = 1);
			sprite(fan::camera* camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size = 0, f_t transparency = 1);

		};

		struct sprite_vector : public fan_2d::sprite_vector {

			sprite_vector(fan::camera* camera, const std::string& path);
			sprite_vector(fan::camera* camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0);

		};

		namespace font_properties {

			inline f_t new_line(80);

			inline fan::color default_text_color(1);

			inline f_t space_width(15);

			inline f_t get_new_line(f_t font_size) {
				return new_line * font_size;
			}

		}

		inline std::unordered_map<fan::window_t, uint_t> current_focus;
		inline std::unordered_map<fan::window_t, uint_t> focus_counter;

		class text_renderer : 
			protected fan::basic_shape_color_vector_vector<4>, 
			protected fan::basic_shape_color_vector_vector<4, GL_SHADER_STORAGE_BUFFER>,
			public fan::texture_handler,
			public fan::vao_handler,
			public fan::glsl_location_handler<3, fan::opengl_buffer_type::buffer_object>,
			public fan::glsl_location_handler<1, fan::opengl_buffer_type::buffer_object>,
			public fan::glsl_location_handler<2, fan::opengl_buffer_type::buffer_object>{
		public:

			using text_color_t = fan::basic_shape_color_vector_vector<4>;
			using outline_color_t = fan::basic_shape_color_vector_vector<4, GL_SHADER_STORAGE_BUFFER>;

			using font_sizes_ssbo_t = fan::glsl_location_handler<3, fan::opengl_buffer_type::buffer_object>;

			using vertex_vbo_t = fan::glsl_location_handler<1, fan::opengl_buffer_type::buffer_object>;
			using texture_vbo_t = fan::glsl_location_handler<2, fan::opengl_buffer_type::buffer_object>;

			text_renderer(fan::camera* camera);
			text_renderer(fan::camera* camera, const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color = fan::color(-1, -1, -1, 0), bool queue = false);

			text_renderer(const text_renderer& tr);
			text_renderer(text_renderer&& tr);

			text_renderer& operator=(const text_renderer& tr);
			text_renderer& operator=(text_renderer&& tr);

			fan::vec2 get_position(uint_t i) const;
			
			void set_position(uint_t i, const fan::vec2& position, bool queue = false);

			f_t get_font_size(uint_t i) const;
			void set_font_size(uint_t i, f_t font_size, bool queue = false);
			void set_text(uint_t i, const fan::fstring& text, bool queue = false);
			void set_text_color(uint_t i, const fan::color& color, bool queue = false);
			void set_outline_color(uint_t i, const fan::color& color, bool queue = false);

			fan::io::file::font_t get_letter_info(fan::fstring::value_type c, f_t font_size) const;
			fan::vec2 get_text_size(uint_t i) const {
				return get_text_size(get_text(i), get_font_size(i));
			}

			f_t get_longest_text() const {
				
				f32_t longest = -fan::inf;

				for (uint_t i = 0; i < this->size(); i++) {
					longest = std::max(longest, get_text_size(i).x);
				}

				return longest;
			}

			f_t get_highest_text() const {

				f32_t highest = -fan::inf;

				for (uint_t i = 0; i < this->size(); i++) {
					highest = std::max(highest, get_text_size(i).y);
				}

				return highest;
			}

			fan::vec2 get_text_size(const fan::fstring& text, f_t font_size) const;

			f_t get_lowest(f_t font_size) const;
			f_t get_highest(f_t font_size) const;

			f_t get_highest_size(f_t font_size) const;
			f_t get_lowest_size(f_t font_size) const;

			// i = string[i], j = string[i][j] (fan::fstring::value_type)
			fan::color get_color(uint_t i, uint_t j = 0) const;

			fan::fstring get_text(uint_t i) const;

			f_t convert_font_size(f_t font_size) const;

			void free_queue();
			void insert(uint_t i, const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color = fan::uninitialized, bool queue = false);
			void push_back(const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color = fan::uninitialized, bool queue = false);

			void draw() const;

			void erase(uint_t i, bool queue = false);

			uint_t size() const;

			fan::io::file::font_info m_font_info;

			fan::camera* m_camera;

		private:

			void initialize_buffers();

			uint_t get_character_offset(uint_t i, bool special);
			
			std::vector<fan::vec2> get_vertices(uint_t i);
			std::vector<fan::vec2> get_texture_coordinates(uint_t i);

			void load_characters(uint_t i, fan::vec2 position, const fan::fstring& text, bool edit, bool insert);

			void edit_letter_data(uint_t i, uint_t j, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f_t converted_font_size);
			void insert_letter_data(uint_t i, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f_t converted_font_size);
			void write_letter_data(uint_t i, const fan::fstring::value_type letter, const fan::vec2& position, int& advance, f_t converted_font_size);

			void write_vertices();
			void write_texture_coordinates();
			void write_font_sizes();

			void write_data();

			fan::shader m_shader;

			fan::vec2ui m_original_image_size;

			std::vector<fan::fstring> m_text;

			std::vector<fan::vec2> m_position;

			std::vector<std::vector<f32_t>> m_font_size;
			std::vector<std::vector<fan::vec2>> m_vertices;
			std::vector<std::vector<fan::vec2>> m_texture_coordinates;

		};

		namespace text_box_properties {

			inline int blink_speed(500); // ms
			inline fan::color cursor_color(fan::colors::white);

		}

		template <typename T>
		class base_box {
		public:

			using value_type = T;

			base_box(fan::camera* camera)
				: m_focus_begin(0), m_focus_end(fan::uninitialized), m_tr(camera), m_rv(camera) { }

			base_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& text_color)
				: m_focus_begin(0), m_focus_end(fan::uninitialized) , m_tr(camera, text, position, text_color, font_size), m_rv(camera){ }

			f_t get_highest(f_t font_size) const {
				return m_tr.get_highest(font_size);
			}

			f_t get_lowest(f_t font_size) const {
				return m_tr.get_lowest(font_size);
			}

			fan::fstring get_text(uint_t i) const {
				return m_tr.get_text(i);
			}

			fan::vec2 get_position(uint_t i) const {
				return m_rv.get_position(i);
			}

			fan::vec2 get_size(uint_t i) const {
				return m_rv.get_size(i);
			}

			void set_text(uint_t i, const fan::fstring& text, bool queue = false) {
				m_tr.set_text(i, text, queue);
			}

			fan::color get_box_color(uint_t i) const {
				return m_rv.get_color(i);
			}

			void set_box_color(uint_t i, const fan::color& color, bool queue = false) {
				m_rv.set_color(i, color, queue);
			}

			fan::color get_text_color(uint_t i) const {
				return m_tr.get_color(i);
			}

			void set_text_color(uint_t i, const fan::color& color, bool queue = false) {
				m_tr.set_text_color(i, color, queue);
			}

			f_t get_font_size(uint_t i) const {
				return m_tr.get_font_size(i);
			}

			void draw() {
				m_rv.draw();
				m_tr.draw();
			}

			void erase(uint_t i, bool queue = false) {
				base_box<T>::m_tr.erase(i, queue);
				base_box<T>::m_rv.erase(i, queue);
				base_box<T>::m_new_lines.erase(base_box<T>::m_new_lines.begin() + i);
			}

			uint_t get_focus_id(uint_t i) const {
				return m_focus_id[i];
			}

			void set_focus_id(uint_t i, uint_t id) {
				m_focus_id[i] = id;
			}

			void set_focus_begin(uint_t begin) {
				m_focus_begin = begin;
			}

			void set_focus_end(uint_t end) {
				m_focus_end = end;
			}

		public:

			std::vector<uint_t> m_focus_id;

			uint_t m_focus_begin;
			uint_t m_focus_end;

			fan_2d::gui::text_renderer m_tr;
			T m_rv;

			std::vector<int64_t> m_new_lines;

			std::vector<fan::vec2> m_border_size;

		};

		template <typename T>
		class basic_sized_text_box : public base_box<T> {
		public:

			basic_sized_text_box(fan::camera* camera) 
				: base_box<T>(camera) { }

			basic_sized_text_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& text_color) 
				: base_box<T>(camera, text, font_size, position, text_color) { }

			fan::vec2 get_updated_box_size(uint_t i) {

				const f_t font_size = base_box<T>::m_tr.get_font_size(i);

				f_t h = font_size + (std::abs(base_box<T>::m_tr.get_highest(font_size) + base_box<T>::m_tr.get_lowest(font_size)));

				if (base_box<T>::m_new_lines[i]) {
					h += fan_2d::gui::font_properties::new_line * base_box<T>::m_tr.convert_font_size(font_size) * base_box<T>::m_new_lines[i];
				}

				return m_size[i] + base_box<T>::m_border_size[i];
			}

			// center, left
			void set_position(uint_t i, const fan::vec2& position, bool queue = false) {
				base_box<T>::m_rv.set_position(i, position, queue);
				auto h = (std::abs(this->get_highest(base_box<T>::m_tr.get_font_size(i)) + this->get_lowest(base_box<T>::m_tr.get_font_size(i)))) / 2;
				base_box<T>::m_tr.set_position(i, fan::vec2(position.x, position.y + m_size[i].y * 0.5 - h) +base_box<T>::m_border_size[i] * 0.5, queue);
			}

			/*void set_font_size(uint_t i, f_t font_size, bool queue = false) {
			base_box<T>::m_tr.set_font_size(i, font_size);

			auto h = (std::abs(this->get_highest(base_box<T>::get_font_size(i)) - this->get_lowest(base_box<T>::get_font_size(i)))) * 0.5;

			base_box<T>::m_tr.set_position(i, fan::vec2(base_box<T>::m_rv.get_position(i).x + m_border_size[i].x * 0.5, base_box<T>::m_rv.get_position(i).y + h + m_border_size[i].y * 0.5));

			update_box(i, queue);
			}*/

			uint_t size() const {
				return base_box<T>::m_tr.size();
			}

			void erase(uint_t i, bool queue = false) {
				base_box<T>::erase(i, queue);
				base_box<T>::m_border_size.erase(base_box<T>::m_border_size.begin() + i);
			}

			fan::vec2 get_border_size(uint_t i) const {
				return base_box<T>::m_border_size[i];
			}

			/*void erase(uint_t begin, uint_t end) {

			}*/

			void update_box_size(uint_t i) {
				
				//base_box<T>::m_rv.set_size(i, get_updated_box_size(i, border_size));
			}

		protected:

			std::vector<fan::vec2> m_size;

		};

		template <typename T>
		class basic_text_box : public base_box<T> {
		public:

			basic_text_box(fan::camera* camera) 
				: base_box<T>(camera) { }

			basic_text_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& text_color) 
				: base_box<T>(camera, text, font_size, position, text_color) { }

			fan::vec2 get_updated_box_size(uint_t i) {

				const f_t font_size = base_box<T>::m_tr.get_font_size(i);

				f_t h = font_size + (std::abs(base_box<T>::m_tr.get_highest(font_size) + base_box<T>::m_tr.get_lowest(font_size)));

				if (base_box<T>::m_new_lines[i]) {
					h += fan_2d::gui::font_properties::new_line * base_box<T>::m_tr.convert_font_size(font_size) * base_box<T>::m_new_lines[i];
				}

				return (base_box<T>::m_tr.get_text(i).empty() ? fan::vec2(0, h) : fan::vec2(base_box<T>::m_tr.get_text_size(base_box<T>::m_tr.get_text(i), font_size).x, h)) + base_box<T>::m_border_size[i];
			}

			void set_position(uint_t i, const fan::vec2& position, bool queue = false) {
				base_box<T>::m_rv.set_position(i, position, queue);
				auto h = (std::abs(this->get_highest(base_box<T>::m_tr.get_font_size(i)) + this->get_lowest(base_box<T>::m_tr.get_font_size(i)))) / 2;
				base_box<T>::m_tr.set_position(i, fan::vec2(position.x, position.y + h) + base_box<T>::m_border_size[i] * 0.5, queue);
			}

			/*void set_font_size(uint_t i, f_t font_size, bool queue = false) {
				base_box<T>::m_tr.set_font_size(i, font_size);

				auto h = (std::abs(this->get_highest(base_box<T>::get_font_size(i)) - this->get_lowest(base_box<T>::get_font_size(i)))) * 0.5;

				base_box<T>::m_tr.set_position(i, fan::vec2(base_box<T>::m_rv.get_position(i).x + m_border_size[i].x * 0.5, base_box<T>::m_rv.get_position(i).y + h + m_border_size[i].y * 0.5));

				update_box(i, queue);
			}*/

			uint_t size() const {
				return base_box<T>::m_tr.size();
			}

			void erase(uint_t i, bool queue = false) {
				base_box<T>::erase(i, queue);
				base_box<T>::m_border_size.erase(base_box<T>::m_border_size.begin() + i);
			}

			fan::vec2 get_border_size(uint_t i) const {
				return base_box<T>::m_border_size[i];
			}

			/*void erase(uint_t begin, uint_t end) {

			}*/

			void update_box_size(uint_t i) {
				base_box<T>::m_rv.set_size(i, get_updated_box_size(i));
			}

		};

		template <typename T>
		class text_box_mouse_input {
		public:

			text_box_mouse_input(T& rv, std::vector<uint_t>& focus_id) : m_rv(rv), m_focus_id(focus_id) {}

			bool inside(uint_t i) const {
				return m_rv.inside(i);
			}

			void on_touch(std::function<void(uint_t j)> function) {
				m_on_touch = function;

				m_rv.m_camera->m_window->add_mouse_move_callback([&] {
					for (uint_t i = 0; i < m_rv.size(); i++) {
						if (m_rv.inside(i)) {
							m_on_touch(i);
						}
					}
				});
			}

			void on_exit(std::function<void(uint_t j)> function) {
				m_on_exit = function;

				m_rv.m_camera->m_window->add_mouse_move_callback([&] {
					for (uint_t i = 0; i < m_rv.size(); i++) {
						if (!inside(i)) {
							m_on_exit(i);
						}
					}
				});
			}

			void on_click(std::function<void(uint_t i)> function, fan::input key = fan::mouse_left) {
				if (m_on_click) {
					m_on_click = function;
				}
				else {
					m_on_click = function;

					m_rv.m_camera->m_window->add_key_callback(key, [&] {
						for (uint_t i = 0; i < m_rv.size(); i++) {
							if (m_rv.inside(i)) {

								current_focus[m_rv.m_camera->m_window->get_handle()] = m_focus_id[i];

								m_on_click(i);
								break;
							}
						}
					});
				}
			}

			void on_release(std::function<void(uint_t i)> function, uint16_t key = fan::mouse_left) {
				m_on_release = function;

				m_rv.m_camera->m_window->add_key_callback(key, [&] {
					for (uint_t i = 0; i < m_rv.size(); i++) {
						if (inside(i)) {
							m_on_release(i);
						}
					}
				}, true);
			}

		protected:

			std::function<void(uint_t i)> m_on_touch;
			std::function<void(uint_t i)> m_on_exit;
			std::function<void(uint_t i)> m_on_click;
			std::function<void(uint_t i)> m_on_release;

		protected:
			
			T& m_rv;

			std::vector<uint_t>& m_focus_id;

		};

		template <typename box_type>
		class text_box_keyboard_input : public box_type {
		public:

			text_box_keyboard_input(fan::camera* camera)
				:	box_type(camera), m_text_visual_input(camera) {

				box_type::m_tr.m_camera->m_window->add_keys_callback([&](fan::fstring::value_type key) {
					for (uint_t j = 0; j < m_callable.size(); j++) {
						bool break_loop = handle_input(m_callable[j], key);
						if (break_loop) {
							break;
						}
					}
				});
			}

			text_box_keyboard_input(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& text_color) 
				:	box_type(camera, text, font_size, position, text_color), m_text_visual_input(camera) {

				box_type::m_tr.m_camera->m_window->add_keys_callback([&](fan::fstring::value_type key) {
					for (uint_t j = 0; j < m_callable.size(); j++) {
						bool break_loop = handle_input(m_callable[j], key);
						if (break_loop) {
							break;
						}
					}
				});
			}

			void set_text(uint_t i, const fan::fstring& text, bool queue = false) {
				box_type::set_text(i, text, queue);
				update_box(i);
			}

			void set_cursor_visible(uint_t i, bool state = true) {
				m_text_visual_input.m_visible[i] = state;
				m_text_visual_input.m_timer[i].restart();
				update_cursor_position(i);
			}

			void push_back(uint_t i) {
				
				const auto& str = box_type::m_tr.get_text(i);

				m_text_visual_input.m_cursor.push_back(fan::mat2(), text_box_properties::cursor_color);
				m_text_visual_input.m_timer.emplace_back(fan::timer<>(fan::timer<>::start(), text_box_properties::blink_speed));
				m_text_visual_input.m_visible.emplace_back(false);
				m_line_offset.resize(i + 1);
				m_line_offset[i].emplace_back(0);

				m_characters_per_line.resize(i + 1);

				uint_t new_lines = 0;
				uint_t characters_per_line = 0;

				for (std::size_t j = 0; j < str.size(); j++) {
					characters_per_line++;
					if (str[j] == '\n') {
						m_characters_per_line[i].emplace_back(characters_per_line);
						m_line_offset[i].emplace_back(characters_per_line);
						new_lines++;
						characters_per_line = 0;
					}
				}

				m_characters_per_line[i].emplace_back(characters_per_line);
				m_current_character.emplace_back(characters_per_line);
				
				text_box_keyboard_input::base_box::m_new_lines.emplace_back(new_lines);
				m_current_cursor_line.emplace_back(new_lines);
			}

			// returns begin and end of cursor points
			fan::mat2 get_cursor_position(uint_t i, uint_t beg = fan::uninitialized, uint_t n = fan::uninitialized) const {

				const auto& str = box_type::m_tr.get_text(i);

				const auto font_size = box_type::m_tr.get_font_size(i);

				auto converted = this->box_type::m_tr.convert_font_size(font_size);

				f_t x = 0;
				f_t y = 0;

				const std::size_t b = beg == (std::size_t)-1 ? 0 : beg;

				const std::size_t n_ = n == (std::size_t)-1 ? str.size() : n;
				
				for (std::size_t j = b; j < b + n_ && j < str.size(); j++) {

					if (str[j] == '\n') {
						continue;
					}

					auto found = box_type::m_tr.m_font_info.m_font.find(str[j]);
					if (found != box_type::m_tr.m_font_info.m_font.end()) {
						x += found->second.m_advance * converted;
					}

				}
				
				const f_t new_line_size = box_type::m_rv.get_size(i).y - box_type::m_border_size[i].y;

				y += m_current_cursor_line[i] * new_line_size;

				return fan::mat2(
					fan::vec2(x, y), 
					fan::vec2(x, y + new_line_size)
				) + box_type::m_rv.get_position(i) +  box_type::m_border_size[i] * 0.5;
			}

			void update_cursor_position(uint_t i) {
				if (i >= m_text_visual_input.m_cursor.size()) {
					return;
				}

				const fan::mat2 cursor_position = this->get_cursor_position(i, m_line_offset[i][m_current_cursor_line[i]], m_current_character[i]);

				m_text_visual_input.m_cursor.set_position(i, cursor_position);
			}

			void draw() {

				if (!m_text_visual_input.m_cursor.size()) {
					return;
				}

				bool found = false;
				uint_t draw_id = 0;
				
				for (uint_t i = 0; i < box_type::m_focus_id.size(); i++) {
					if (box_type::m_focus_id[i] == current_focus[box_type::m_tr.m_camera->m_window->get_handle()]) {
						found = true;
						draw_id = i;
						break;
					}
				}

				auto cfound = std::find(m_callable.begin(), m_callable.end(), draw_id);

				if (!found || cfound == m_callable.end()) {
					return;
				}

				//for (uint_t i = 0; i < m_text_visual_input.m_cursor.size(); i++) {

					if (m_text_visual_input.m_timer[draw_id].finished()) {
						m_text_visual_input.m_visible[draw_id] = !m_text_visual_input.m_visible[draw_id];
						m_text_visual_input.m_timer[draw_id].restart();
					}
				//}

				//if (m_callable.size() == m_text_visual_input.m_cursor.size()) {
				//	if (m_text_visual_input.m_visible[draw_id]) {
				//		m_text_visual_input.m_cursor.draw(draw_id);
				//	}	
				//}
				//else { // in case we dont want to draw input for some window
				//	for (std::size_t i = 0; i < m_callable.size(); i++) {
						if (m_text_visual_input.m_visible[draw_id]) {
							m_text_visual_input.m_cursor.draw(draw_id);
						}
				/*	}
				}*/
			}

			void update_box(uint_t i) {
				m_characters_per_line[i].clear();
				m_line_offset[i].clear();

				m_line_offset[i].emplace_back(0);
				m_characters_per_line[i].clear();

				uint_t new_lines = 0;
				uint_t characters_per_line = 0;

				auto str = box_type::m_tr.get_text(i);

				for (std::size_t j = 0; j < str.size(); j++) {
					characters_per_line++;
					if (str[j] == '\n') {
						m_characters_per_line[i].emplace_back(characters_per_line);
						m_line_offset[i].emplace_back(characters_per_line);
						new_lines++;
						characters_per_line = 0;
					}
				}

				m_characters_per_line[i].emplace_back(characters_per_line);
				m_current_character[i] = characters_per_line;

				text_box_keyboard_input::base_box::m_new_lines[i] = new_lines;
				m_current_cursor_line[i] = new_lines;

				box_type::update_box_size(i);
				update_cursor_position(i);
			}

			bool handle_input(uint_t i, uint_t key) {
				if (current_focus[box_type::m_tr.m_camera->m_window->get_handle()] != box_type::m_focus_id[i]) {
					return false;
				}

				auto str = box_type::m_tr.get_text(i);

				auto current_key = box_type::m_tr.m_camera->m_window->get_current_key();

				switch (current_key) {
					case fan::key_v:
					{
						if (box_type::m_tr.m_camera->m_window->key_press(fan::key_control)) {

							str = fan::io::get_clipboard_text();

							box_type::m_tr.set_text(i, str);

							update_box(i);
						}
						else {
							goto add_key;
						}

						break;
					}
					case fan::key_delete: {

						if (str.size()) {

							if (str[m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i]] == '\n') {
								m_characters_per_line[i][m_current_cursor_line[i]] += m_characters_per_line[i][m_current_cursor_line[i] + 1] - 1;
								m_characters_per_line[i].erase(m_characters_per_line[i].begin() + m_current_cursor_line[i] + 1);
								m_line_offset[i][m_current_cursor_line[i] + 1] = m_line_offset[i][m_current_cursor_line[i]];
								m_line_offset[i].erase(m_line_offset[i].begin() + m_current_cursor_line[i]);
								text_box_keyboard_input::base_box::m_new_lines[i]--;
							}
							else {
								m_characters_per_line[i][m_current_cursor_line[i]]--;
							}

							if ((uint_t)(m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i]) >= str.size()) {
								break;
							}

							str.erase(str.begin() + m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i]);

							box_type::m_tr.set_text(i, str);

							for (int j = m_current_cursor_line[i] + 1; j <= text_box_keyboard_input::base_box::m_new_lines[i]; j++) {
								m_line_offset[i][j]--;
							}

							update_cursor_position(i);
							box_type::update_box_size(i);
						}

						break;
					}
					case fan::key_backspace:
					{
						if (m_current_character[i] || m_current_cursor_line[i]) {
							
							m_current_character[i]--;

							str.erase(str.begin() + m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i]);

							box_type::m_tr.set_text(i, str);

							// previous line
							if (m_current_cursor_line[i] && m_current_character[i] == -1) {
								m_current_character[i] = m_characters_per_line[i][m_current_cursor_line[i] - 1] - 1;
								m_characters_per_line[i][m_current_cursor_line[i] - 1] += m_characters_per_line[i][m_current_cursor_line[i]];
								m_characters_per_line[i].erase(m_characters_per_line[i].begin() + m_current_cursor_line[i]);
								m_line_offset[i].erase(m_line_offset[i].begin() + m_current_cursor_line[i]);
								m_current_cursor_line[i]--;
								text_box_keyboard_input::base_box::m_new_lines[i]--;
							}
							else if (m_current_character[i] == -1) {
								m_current_character[i] = 0;
							}

							if (m_characters_per_line[i][m_current_cursor_line[i]]) {
								m_characters_per_line[i][m_current_cursor_line[i]]--;
							}

							if (m_current_cursor_line[i] || m_characters_per_line[i][m_current_cursor_line[i]]) {
								for (int j = m_current_cursor_line[i] + 1; j <= text_box_keyboard_input::base_box::m_new_lines[i]; j++) {
									m_line_offset[i][j]--;
								}
							}

							update_cursor_position(i);
							box_type::update_box_size(i);
						}

						break;
					}
					case fan::key_left:
					{

						m_text_visual_input.m_visible[i] = true;

						bool go_once = false;

						if (box_type::m_tr.m_camera->m_window->key_press(fan::key_control)) {

							std::size_t found = -1;

							const auto offset = m_line_offset[i][m_current_cursor_line[i]];

							if (m_current_character[i] - 1 >= 0 && (uint_t)offset < str.size()) {
								auto str_ = str.substr(offset, m_current_character[i] ? m_current_character[i] - 1 : m_current_character[i]);
								found = str_.find_last_of(L' ');
							}

							if (found != std::string::npos) {
								m_current_character[i] = found + 1;

								if (str[m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i]] == ' ') {
									while (str[m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i] - 1] == ' ') {
										m_current_character[i]--;
									}
									while (m_current_character[i] > 0 && str[m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i] - 1] != ' ') {
										m_current_character[i]--;
									}
								}
							}
							else if (
								m_current_character[i] - 1 <= -1) {
								go_once = true;
							}
							else {
							 	m_current_character[i] = 0;
							}
						}
						else {
							go_once = true;
						}

						if (go_once) {
							m_current_character[i]--;
							if (m_current_cursor_line[i] && m_current_character[i] == -1) {
								m_current_cursor_line[i]--;
								m_current_character[i] = m_characters_per_line[i][m_current_cursor_line[i]] - 1;
							}
							else if (m_current_character[i] == -1) {
								m_current_character[i] = 0;
							}
						}

						update_cursor_position(i);

						break;
					}
					case fan::key_right: {

						m_text_visual_input.m_visible[i] = true;

						if (m_current_cursor_line[i] == *(text_box_keyboard_input::base_box::m_new_lines.end() - 1) && m_characters_per_line[i][m_current_cursor_line[i]] <= m_current_character[i]) {
							break;
						}

						bool go_once = false;

						if (box_type::m_tr.m_camera->m_window->key_press(fan::key_control)) {

							std::size_t found = -1;

							const auto offset = m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i] + 1;

							if ((uint_t)offset < str.size()) {
								found = str.substr(offset, m_characters_per_line[i][m_current_cursor_line[i]] - m_current_character[i]).find_first_of(L' ');
							}

							if (found != std::string::npos) {
								m_current_character[i] = m_current_character[i] + found + 1;
								if (str[m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i] - 1] == ' ') {
									while (str[m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i]] == ' ') {
										m_current_character[i]++;
									}
									while (m_current_character[i] < m_characters_per_line[i][m_current_cursor_line[i]] && str[m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i]] != ' ') {
										m_current_character[i]++;
									}
								}
							}
							else if (m_current_character[i] + 1 >= m_characters_per_line[i][m_current_cursor_line[i]]) {
								go_once = true;
							}
							else {
								m_current_character[i] = m_characters_per_line[i][m_current_cursor_line[i]];
							}
						}
						else {
							go_once = true;
						}

						if (go_once) {
							const auto offset = m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i];

							if (m_current_character[i] >= m_characters_per_line[i][m_current_cursor_line[i]] || (*(str.begin() + offset) == '\n')) {
								m_current_character[i] = 0;
								m_current_cursor_line[i]++;
							}
							else {
								m_current_character[i] = std::clamp(++m_current_character[i], (int64_t)0, (int64_t)m_characters_per_line[i][m_current_cursor_line[i]]);
							}
						}

						update_cursor_position(i);

						break;
					}
					case fan::key_home:
					{
						m_text_visual_input.m_visible[i] = true;

						if (box_type::m_tr.m_camera->m_window->key_press(fan::key_control)) {
							m_current_cursor_line[i] = 0;
						}

						m_current_character[i] = 0;

						update_cursor_position(i);

						break;
					}
					case fan::key_end:
					{
						m_text_visual_input.m_visible[i] = true;

						if (box_type::m_tr.m_camera->m_window->key_press(fan::key_control)) {
							m_current_cursor_line[i] = text_box_keyboard_input::base_box::m_new_lines[i];
						}

						auto b = text_box_keyboard_input::base_box::m_new_lines[i] && m_characters_per_line[i][m_current_cursor_line[i]] && str[m_line_offset[i][m_current_cursor_line[i]] + m_characters_per_line[i][m_current_cursor_line[i]] - 1] == '\n';

						m_current_character[i] = m_characters_per_line[i][m_current_cursor_line[i]] - b;

						update_cursor_position(i);

						break;
					}
					case fan::key_up:
					{
						m_text_visual_input.m_visible[i] = true;

						if (m_current_cursor_line[i] > 0) {

							f_t fclosest = fan::inf;

							f_t current = 0;

							for (int j = 0; j < m_current_character[i]; j++) {
								auto c = *(str.begin() + m_line_offset[i][m_current_cursor_line[i]] + j);

								if (c == '\n') {
									continue;
								}

								auto l_info = box_type::m_tr.get_letter_info(c, box_type::m_tr.get_font_size(i));
								current += l_info.m_advance;
							}

							m_current_cursor_line[i]--;

							f_t new_current = 0;
							std::size_t iclosest = 0;

							if (fan::distance(new_current, current) < fclosest) {
								fclosest = fan::distance(new_current, current);
								iclosest = 0;
							}

							for (int j = m_line_offset[i][m_current_cursor_line[i]]; j < m_line_offset[i][m_current_cursor_line[i]] + m_characters_per_line[i][m_current_cursor_line[i]]; j++) {

								auto c = str[j];

								if (c == '\n') {
									continue;
								}

								auto l_info = box_type::m_tr.get_letter_info(c, box_type::m_tr.get_font_size(i));

								new_current += l_info.m_advance;

								if (fan::distance(new_current, current) < fclosest) {
									fclosest = fan::distance(new_current, current);
									iclosest = j - m_line_offset[i][m_current_cursor_line[i]];
								}
							}

							if (!new_current) {
								m_current_character[i] = m_characters_per_line[i][m_current_cursor_line[i]];
								if (str[m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i] - (bool)(m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i])] == '\n') {
									m_current_character[i] = 0;
								}
							}
							else {
								m_current_character[i] = iclosest + (bool)m_current_character[i];
							}

							update_cursor_position(i);
						}

						break;
					}
					case fan::key_down:
					{
						m_text_visual_input.m_visible[i] = true;

						if (m_current_cursor_line[i] < text_box_keyboard_input::base_box::m_new_lines[i]) {

							f_t fclosest = fan::inf;
							std::size_t iclosest = 0;

							f_t current = 0;

							for (int j = 0; j < m_current_character[i]; j++) {

								auto c = *(str.begin() + m_line_offset[i][m_current_cursor_line[i]] + j);

								if (c == '\n') {
									continue;
								}

								auto l_info = box_type::m_tr.get_letter_info(c, box_type::m_tr.get_font_size(i));
								current += l_info.m_advance;
							}

							m_current_cursor_line[i]++;

							f_t new_current = 0;

							if (fan::distance(new_current, current) < fclosest) {
								fclosest = fan::distance(new_current, current);
								iclosest = 0;
							}

							for (int j = 0; j < m_characters_per_line[i][m_current_cursor_line[i]]; j++) {
								auto c = *(str.begin() + m_line_offset[i][m_current_cursor_line[i]] - 1 + j);

								if (c == '\n') {
									continue;
								}

								auto l_info = box_type::m_tr.get_letter_info(c, box_type::m_tr.get_font_size(i));

								new_current += l_info.m_advance;

								if (fan::distance(new_current, current) < fclosest) {
									fclosest = fan::distance(new_current, current);
									iclosest = j;
								}
							}

							if (!new_current) {
								m_current_character[i] = m_characters_per_line[i][m_current_cursor_line[i]];
								if (str[m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i] - 1] == '\n') {
									m_current_character[i] = 0;
								}
							}
							else {
								m_current_character[i] = iclosest;
							}


							update_cursor_position(i);
						}

						break;
					}
					case fan::key_enter: {

						/*str.insert(str.begin() + m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i], '\n');

						const auto characters_left = m_characters_per_line[i][m_current_cursor_line[i]] - m_current_character[i];

						m_line_offset[i].insert(m_line_offset[i].begin() + m_current_cursor_line[i] + 1, (m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i] + 1));

						m_characters_per_line[i][m_current_cursor_line[i]] -= characters_left;

						m_characters_per_line[i][m_current_cursor_line[i]]++;

						base_box::m_new_lines[i]++;
						m_current_cursor_line[i]++;
						m_current_character[i] = 0;

						m_characters_per_line[i].insert(m_characters_per_line[i].begin() + m_current_cursor_line[i], characters_left);

						box_type::m_tr.set_text(i, str);

						update_box_size(i, border_size);
						update_cursor_position(i, border_size);

						for (int j = m_current_cursor_line[i] + 1; j <= base_box::m_new_lines[i]; j++) {
							m_line_offset[i][j]++;
						}*/

						break;
					}
					case fan::key_tab:
					{
						m_text_visual_input.m_visible[i] = false;

						// begin not working properly yet
						int64_t min = box_type::m_focus_begin;
						int64_t max = (box_type::m_focus_end == (uint_t)fan::uninitialized ? m_text_visual_input.m_visible.size() : box_type::m_focus_end) - box_type::m_focus_begin;

						if (box_type::m_tr.m_camera->m_window->key_press(fan::key_shift)) {

							m_text_visual_input.m_visible[(fan::modi((int64_t)(i - 1), max)) + min] = true;
							m_text_visual_input.m_timer[(fan::modi((int64_t)(i - 1), max)) + min].restart();
							current_focus[box_type::m_tr.m_camera->m_window->get_handle()] = box_type::m_focus_id[(fan::modi((int64_t)(i - 1), max)) + min];
							update_cursor_position((fan::modi((int64_t)(i - 1), max)) + min);
						}
						else {
							m_text_visual_input.m_visible[((i + 1) % max) + min] = true;
							m_text_visual_input.m_timer[((i + 1) % max) + min].restart();
							current_focus[box_type::m_tr.m_camera->m_window->get_handle()] = box_type::m_focus_id[((i + 1) % max) + min];
							update_cursor_position(((i + 1) % max) + min);
						}

						current_focus[box_type::m_tr.m_camera->m_window->get_handle()] = fan::modi(
							(int64_t)current_focus[box_type::m_tr.m_camera->m_window->get_handle()], 
							(int64_t)focus_counter[box_type::m_tr.m_camera->m_window->get_handle()]
						);

						return true;
					}
					default:
					{
						add_key:

						for (uint_t j = 0; j < box_type::m_tr.m_camera->m_window->m_key_exceptions.size(); j++) {
							if (current_key == box_type::m_tr.m_camera->m_window->m_key_exceptions[j]) {
								return false;
							}
						}

						str.insert(str.begin() + m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i], key);
						m_characters_per_line[i][m_current_cursor_line[i]]++;
						m_current_character[i]++;

						for (int j = m_current_cursor_line[i] + 1; j <= text_box_keyboard_input::base_box::m_new_lines[i]; j++) {
							m_line_offset[i][j]++;
						}

						text_box_keyboard_input::base_box::m_tr.set_text(i, str);
						box_type::update_box_size(i);

						update_cursor_position(i);
					}
				}

				return false;
			}

			void get_mouse_cursor(uint_t i, const fan::vec2& box_position, const fan::vec2& box_size) {

				const auto& str = box_type::m_tr.get_text(i);

				m_text_visual_input.m_visible[i] = true;

				fan::vec2 mouse_position(box_type::m_tr.m_camera->m_window->get_mouse_position());

				const fan::vec2 border_size = box_type::get_border_size(i);

				mouse_position.x = fan::clamp(mouse_position.x, box_position.x + border_size.x * 0.5f, mouse_position.x);
				mouse_position.y = fan::clamp(mouse_position.y, box_position.y + border_size.y * 0.5f, mouse_position.y);

				f_t fclosest = fan::inf;
				int64_t iclosest = 0;

				f_t current = 0;

				for (int j = 0; j < m_characters_per_line[i][m_current_cursor_line[i]] + 1; j++) {

					fan::fstring::value_type c;

					if (j != m_characters_per_line[i][m_current_cursor_line[i]]) {
						c = *(str.begin() + m_line_offset[i][m_current_cursor_line[i]] + j);
					}
					else {
						c = ' ';
					}

					if (c == '\n') {
						continue;
					}

					auto l_info = box_type::m_tr.get_letter_info(c, box_type::m_tr.get_font_size(i));
					current += l_info.m_advance;

					if (fan::distance(mouse_position.x - (box_position.x + border_size.x * 0.5), current) < fclosest) {
						fclosest = fan::distance(mouse_position.x - (box_position.x + border_size.x * 0.5), current);
						iclosest = j;
					}
				}

				if (fan::distance(mouse_position.x, box_position.x + border_size.x * 0.5) < fclosest) {
					iclosest = -1;
				}

				m_current_cursor_line[i] = (mouse_position.y - (box_position.y + border_size.y * 0.5)) / fan_2d::gui::font_properties::get_new_line(box_type::m_tr.convert_font_size(box_type::m_tr.get_font_size(i)));

				m_current_cursor_line[i] = fan::clamp(m_current_cursor_line[i], (int64_t)0, text_box_keyboard_input::base_box::m_new_lines[i]);

				m_current_character[i] = fan::clamp(iclosest + 1, (int64_t)0, m_characters_per_line[i][m_current_cursor_line[i]]);

				if (m_current_character[i] > 0) {
					if (*(str.begin() + m_line_offset[i][m_current_cursor_line[i]] + m_current_character[i] - 1) == '\n') {
						m_current_character[i] = fan::clamp(m_current_character[i] - 1, (int64_t)0, m_characters_per_line[i][m_current_cursor_line[i]]);
					}
				}

				update_cursor_position(i);
			}

			fan::fstring get_line(uint_t i, uint_t line) {
				const auto& str = box_type::m_tr.get_text(i);

				if (box_type::m_tr.size() < i) {
					return L"";
				}
				else if (line > text_box_keyboard_input::base_box::m_new_lines[i]) {
					return L"";
				}
				return str.substr(m_line_offset[i][line], m_characters_per_line[i][line]);
			}

			void erase(uint_t i, bool queue = false) {
				m_text_visual_input.m_cursor.erase(i, queue);
				m_text_visual_input.m_timer.erase(i);
				m_text_visual_input.m_visible.erase(i);

				m_callable.erase(m_callable.begin() + i);
				m_current_cursor_line.erase(m_current_cursor_line.begin() + i);
				m_current_character.erase(m_current_character.begin() + i);
				m_characters_per_line.erase(m_characters_per_line.begin() + i);
				m_line_offset.erase(m_line_offset.begin() + i);

			}

			void set_input_callback(uint_t i) {
				m_callable.emplace_back(i);
			}

		private:

			struct text_visual_input {

				text_visual_input(fan::camera* camera) : m_cursor(camera) {}
				text_visual_input(fan::camera* camera, const fan::mat2& position, const fan::color& color = text_box_properties::cursor_color) 
						: m_cursor(camera, position, color) {}

				fan_2d::line_vector m_cursor;
				std::vector<fan::timer<>> m_timer;
				std::vector<bool> m_visible;
			};

			text_visual_input m_text_visual_input;

			std::vector<int64_t> m_callable;

			std::vector<int64_t> m_current_cursor_line;
			std::vector<int64_t> m_current_character;
			std::vector<std::vector<int64_t>> m_characters_per_line;
			std::vector<std::vector<int64_t>> m_line_offset;

		};

		struct text_box : 
			public text_box_mouse_input<fan_2d::rectangle_vector>,
			public text_box_keyboard_input<basic_text_box<fan_2d::rectangle_vector>> {

			using value_type = fan_2d::rectangle_vector;

			using basic_box = basic_text_box<value_type>;
			using mouse_input = text_box_mouse_input<value_type>;
			using keyboard_input = text_box_keyboard_input<basic_box>;

			text_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, const fan::color& text_color = fan::colors::white) 
				: mouse_input(basic_box::m_rv, keyboard_input::m_focus_id), keyboard_input(camera, text, font_size, position + border_size / 2, text_color)
			{ 
				camera->m_window->add_resize_callback([&] {
					for (uint_t i = 0; i < basic_box::m_rv.size(); i++) {
						const auto offset = fan_2d::gui::get_resize_movement_offset(camera->m_window);
						basic_box::m_rv.set_position(i, basic_box::m_rv.get_position(i) + offset);
						basic_box::m_tr.set_position(i, basic_box::m_tr.get_position(i) + offset);
						update_cursor_position(i);
					}
				});

				on_click([&](uint_t i) {});

				basic_box::m_border_size.emplace_back(border_size);

				auto h = (std::abs(this->get_highest(get_font_size(0)) + this->get_lowest(get_font_size(0)))) / 2;

				basic_box::m_tr.set_position(0, fan::vec2(position.x + border_size.x * 0.5, position.y + h + border_size.y * 0.5));

				keyboard_input::push_back(basic_box::m_border_size.size() - 1);

				basic_box::m_rv.push_back(position, basic_box::get_updated_box_size(basic_box::m_border_size.size() - 1), box_color);

				keyboard_input::update_box_size(this->size() - 1);
				update_cursor_position(this->size() - 1);

				auto found = focus_counter.find(basic_box::m_tr.m_camera->m_window->get_handle());

				if (found == focus_counter.end()) {
					fan_2d::gui::focus_counter.insert(std::make_pair(basic_box::m_tr.m_camera->m_window->get_handle(), 0));
				}

				auto new_found = focus_counter.find(basic_box::m_tr.m_camera->m_window->get_handle());

				keyboard_input::m_focus_id.emplace_back(new_found->second);
				focus_counter[new_found->first]++;
			}

			void push_back(const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, const fan::color& text_color = fan::colors::white);

			void draw() {
				fan::draw_2d([&] {
					basic_box::draw();
					keyboard_input::draw();
				});
			}

		};

		struct rounded_text_box : 
			public text_box_mouse_input<fan_2d::rounded_rectangle>,
			public text_box_keyboard_input<basic_text_box<fan_2d::rounded_rectangle>> {

			using value_type = fan_2d::rounded_rectangle;

			using basic_box = basic_text_box<value_type>;
			using mouse_input = text_box_mouse_input<value_type>;
			using keyboard_input = text_box_keyboard_input<basic_box>;

			rounded_text_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, f_t radius, const fan::color& text_color = fan::colors::white)
				:	mouse_input(basic_box::m_rv, keyboard_input::m_focus_id), text_box_keyboard_input<basic_text_box<fan_2d::rounded_rectangle>>(camera, text, font_size, position, text_color)
			{
				camera->m_window->add_resize_callback([&] {
					for (uint_t i = 0; i < basic_box::m_rv.size(); i++) {
						const auto offset = fan_2d::gui::get_resize_movement_offset(camera->m_window);
						basic_box::m_rv.set_position(i, basic_box::m_rv.get_position(i) + offset);
						basic_box::m_tr.set_position(i, basic_box::m_tr.get_position(i) + offset);
						update_cursor_position(i);
					}
				});

				mouse_input::on_click([&] (uint_t i) {});

				basic_box::m_border_size.emplace_back(border_size);

				keyboard_input::push_back(basic_box::m_border_size.size() - 1);

				const auto size = basic_box::get_updated_box_size(basic_box::size() - 1);

				auto h = (std::abs(this->get_highest(get_font_size(basic_box::size() - 1)) + this->get_lowest(get_font_size(basic_box::m_border_size.size() - 1)))) / 2;

				basic_box::m_tr.set_position(0, fan::vec2(position.x + border_size.x * 0.5, position.y + h + border_size.y * 0.5));

				basic_box::m_rv.push_back(position, size, radius, box_color);

				keyboard_input::m_focus_id.emplace_back(focus_counter[camera->m_window->get_handle()]++);

				keyboard_input::update_box_size(this->size() - 1);
				update_cursor_position(this->size() - 1);

			}

			void push_back(const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, f_t radius, const fan::color& text_color = fan::colors::white);

			void set_input_callback(uint_t i) {
				keyboard_input::set_input_callback(i);
			}

			void set_position(uint_t i, const fan::vec2& position, bool queue = false) {
				basic_box::set_position(i, position, queue);
				update_cursor_position(i);
			}

			void draw() {
				fan::draw_2d([&] {
					basic_box::draw();
					keyboard_input::draw();
				});
			}

		};

		struct sized_text_box : 
			public text_box_mouse_input<fan_2d::rectangle_vector>,
			public text_box_keyboard_input<basic_sized_text_box<fan_2d::rectangle_vector>> {

			using value_type = fan_2d::rectangle_vector;

			using basic_box = basic_sized_text_box<value_type>;
			using mouse_input = text_box_mouse_input<value_type>;
			using keyboard_input = text_box_keyboard_input<basic_box>;

			sized_text_box(fan::camera* camera) : mouse_input(basic_box::m_rv, keyboard_input::m_focus_id), keyboard_input(camera) { on_click([](uint_t i) {}); }

			sized_text_box(fan::camera* camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::vec2& size, const fan::vec2& border_size, const fan::color& box_color, const fan::color& text_color = fan::colors::white) 
				: mouse_input(basic_box::m_rv, keyboard_input::m_focus_id), keyboard_input(camera, text, font_size, position + border_size / 2, text_color)
			{ 
				camera->m_window->add_resize_callback([&] {
					for (uint_t i = 0; i < basic_box::m_rv.size(); i++) {
						const auto offset = fan_2d::gui::get_resize_movement_offset(keyboard_input::m_tr.m_camera->m_window);
						basic_box::m_rv.set_position(i, basic_box::m_rv.get_position(i) + offset);
						basic_box::m_tr.set_position(i, basic_box::m_tr.get_position(i) + offset);
						update_cursor_position(i);
					}
				});

				on_click([](uint_t i) {});

				m_size.emplace_back(size);

				basic_box::m_border_size.emplace_back(border_size);

				auto h = (std::abs(this->get_highest(get_font_size(0)) + this->get_lowest(get_font_size(0)))) / 2;

				basic_box::m_tr.set_position(0, fan::vec2(position.x + size.x * 0.5 - keyboard_input::m_tr.get_text_size(text, font_size).x * 0.5, position.y + size.y * 0.5 - h) + border_size * 0.5);

				keyboard_input::push_back(basic_box::m_border_size.size() - 1);

				basic_box::m_rv.push_back(position, size + border_size, box_color);

				update_cursor_position(this->size() - 1);

				auto found = focus_counter.find(basic_box::m_tr.m_camera->m_window->get_handle());

				if (found == focus_counter.end()) {
					fan_2d::gui::focus_counter.insert(std::make_pair(basic_box::m_tr.m_camera->m_window->get_handle(), 0));
				}

				auto new_found = focus_counter.find(basic_box::m_tr.m_camera->m_window->get_handle());

				keyboard_input::m_focus_id.emplace_back(new_found->second);
				focus_counter[new_found->first]++;
			}

			void push_back(const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::vec2& size, const fan::vec2& border_size, const fan::color& box_color, const fan::color& text_color = fan::colors::white) {

				m_size.emplace_back(size);

				basic_box::m_border_size.emplace_back(border_size);

				auto h = (std::abs(this->get_highest(font_size) + this->get_lowest(font_size))) / 2;

				basic_box::m_tr.push_back(text, fan::vec2(position.x + size.x * 0.5 - keyboard_input::m_tr.get_text_size(text, font_size).x * 0.5, position.y + size.y * 0.5 - h) + border_size * 0.5, text_color, font_size);

				keyboard_input::push_back(basic_box::m_border_size.size() - 1);

				basic_box::m_rv.push_back(position, size + border_size, box_color);

				update_cursor_position(this->size() - 1);

				auto found = focus_counter.find(basic_box::m_tr.m_camera->m_window->get_handle());

				if (found != focus_counter.end()) {
					keyboard_input::m_focus_id.emplace_back(found->second);
					focus_counter[found->first]++;
				}
				else {
					keyboard_input::m_focus_id.emplace_back(0);
					focus_counter.insert(std::make_pair(keyboard_input::m_tr.m_camera->m_window->get_handle(), 0));
				}

			}

			void draw() {
				fan::draw_2d([&] {
					base_box::draw();
					keyboard_input::draw();
				});
			}

		};

		class basic_selectable_box {
		public:

			basic_selectable_box() : m_selected(fan::uninitialized) { }

			int64_t get_selected() {
				return m_selected;
			}

			void set_selected(int64_t i) {
				m_selected = i;
			}

			/*void color_on_click(uint_t i, const fan::color& color) {
				mouse_input_t::on_click(i, [&] {
					box_t::set_box_color(i, color);
					this->set_selected(i);
				});
			}*/

		private:

		//	mouse_input_t& m_mouse_input;
			//box_t& m_box;

			int64_t m_selected;

		};

		struct selectable_text_box : public text_box, public basic_selectable_box {
			using text_box::text_box;
		};

		struct selectable_rounded_text_box : public rounded_text_box, public basic_selectable_box {
			using rounded_text_box::rounded_text_box;
		};

		struct selectable_sized_text_box : public sized_text_box, public basic_selectable_box {
			using sized_text_box::sized_text_box;
		};

		//class slider : public text_box {
		//public:

		//	slider(fan::camera* camera, f_t min, f_t max, f_t font_size, const fan::vec2& position, const fan::color& slider_color, const fan::color& box_color, const fan::vec2& border_size, const fan::color& text_color = fan::colors::white)
		//		: text_box(camera, fan::to_wstring(min), font_size, position, box_color, border_size, text_color) 
		//	{
		//		const auto size(text_box::m_rv.get_size(text_box::m_rv.size() - 1));

		//		m_min.push_back(min);
		//		m_max.push_back(max);
		//		m_value.push_back(0);

		//		text_box::m_rv.push_back(position + 5, fan::vec2(size.x / 20, size.y - 5 * 2), slider_color);

		//		m_moving.resize(m_moving.size() + 1);

		//	}

		//	f_t get_value(uint_t i) const {
		//		return m_value[i];
		//	}

		//	void set_value(uint_t i, f_t value) {
		//		m_value[i] = value;
		//	}

		//	void move()	{
		//		const bool left_press = m_rv.m_camera->m_window->key_press(fan::mouse_left);

		//		for (uint_t i = 1; i < m_rv.size(); i += 2) {
		//			if (m_rv.inside(i >> 1) && left_press) {
		//				m_moving[i >> 1] = true;
		//			}
		//			else if (!left_press) {
		//				m_moving[i >> 1] = false;
		//			}
		//			if (m_moving[i >> 1]) {
		//				const auto mouse_position(m_tr.m_camera->m_window->get_mouse_position());
		//				const auto s_size(m_rv.get_size(i));
		//				const auto b_size(m_rv.get_size(i >> 1));
		//				const auto position(m_rv.get_position(i >> 1));

		//				auto slider_position = std::clamp((mouse_position.x - s_size.x * 0.5), f_t(position.x + 5), f_t(position.x + b_size.x - s_size.x - 5));

		//				auto min = position.x + 5;
		//				auto max = position.x + b_size.x - s_size.x - 5;
		//				auto n = slider_position;

		//				m_rv.set_position(i, fan::vec2(slider_position, position.y + 5));

		//				this->set_value(i >> 1, ((n - min) / (max - min)) * (m_max[i >> i] - m_min[i >> 1]) + m_min[i >> 1]);

		//				const auto& str = fan::to_wstring(this->get_value(i >> 1));

		//				auto s = m_tr.get_text_size(str, get_font_size(i >> 1));
		//				fan::print(s);
		//				m_tr.set_text(i >> 1, str);

		//			}
		//		}
		//	}

		//private:

		//	std::vector<bool> m_moving;
		//	std::vector<f_t> m_min;
		//	std::vector<f_t> m_value;
		//	std::vector<f_t> m_max;
		//	std::vector<f_t> m_previous;

		//};

	}
}

namespace fan_3d {

	namespace shader_paths {
		constexpr auto triangle_vector_vs("glsl/3D/terrain_generator.vs");
		constexpr auto triangle_vector_fs("glsl/3D/terrain_generator.fs");

		constexpr auto shape_vector_vs("glsl/3D/shape_vector.vs");
		constexpr auto shape_vector_fs("glsl/3D/shape_vector.fs");

		constexpr auto model_vs("glsl/3D/models.vs");
		constexpr auto model_fs("glsl/3D/models.fs");

		constexpr auto animation_vs("glsl/3D/animation.vs");
		constexpr auto animation_fs("glsl/3D/animation.fs");

		constexpr auto skybox_vs("glsl/3D/skybox.vs");
		constexpr auto skybox_fs("glsl/3D/skybox.fs");
		constexpr auto skybox_model_vs("glsl/3D/skybox_model.vs");
		constexpr auto skybox_model_fs("glsl/3D/skybox_model.fs");
	}

	void add_camera_rotation_callback(fan::camera* camera);

	class line_vector : public fan::basic_shape<true, fan::vec3>, public fan::basic_shape_color_vector<true> {
	public:

		line_vector(fan::camera* camera);
		line_vector(fan::camera* camera, const fan::mat2x3& begin_end, const fan::color& color);

		void push_back(const fan::mat2x3& begin_end, const fan::color& color, bool queue = false);

		void draw();

		void set_position(uint_t i, const fan::mat2x3 begin_end, bool queue = false);
		
		void release_queue(bool position, bool color);

	private:

		using line_vector::basic_shape::set_position;
		using line_vector::basic_shape::set_size;

	};

	using triangle_vertices_t = fan::vec3;

	class terrain_generator : public fan::basic_shape_color_vector<true> {
	public:

		terrain_generator(fan::camera* camera, const std::string& path, const f32_t texture_scale, const fan::vec3& position, const fan::vec2ui& map_size, f_t triangle_size, const fan::vec2& mesh_size);
		~terrain_generator();

		void insert(const std::vector<triangle_vertices_t>& vertices, const std::vector<fan::color>& color, bool queue = false);
		void push_back(const triangle_vertices_t& vertices, const fan::color& color, bool queue = false);

		template <uint_t i = uint_t(-1)>
		std::conditional_t<i == -1, std::vector<triangle_vertices_t>, triangle_vertices_t> get_vertices();

		void edit_data(uint_t i, const triangle_vertices_t& vertices, const fan::color& color);

		void release_queue();

		void draw();

		void erase_all();

		uint_t size();

		fan::camera* m_camera;

	private:

		fan::shader m_shader;

		uint32_t m_texture;
		uint32_t m_texture_vbo;
		uint32_t m_vao;
		uint32_t m_vertices_vbo;
		uint32_t m_ebo;
		uint32_t m_triangle_size;
		uint32_t m_normals_vbo;

	//	fan_3d::line_vector lv;

		std::vector<triangle_vertices_t> m_triangle_vertices;
		std::vector<unsigned int> m_indices;
		static constexpr auto m_vertice_size = sizeof(triangle_vertices_t);

	};

	struct plane_corners {
		fan::da_t<f32_t, 3> top_left;
		fan::da_t<f32_t, 3> top_right;
		fan::da_t<f32_t, 3> bottom_left;
		fan::da_t<f32_t, 3> bottom_right;
	};

	struct square_corners {

		plane_corners left;
		plane_corners right;
		plane_corners front;
		plane_corners back;
		plane_corners top;
		plane_corners bottom;
	
	};

	class rectangle_vector : public fan::basic_shape<true, fan::vec3>, public fan::texture_handler {
	public:

		rectangle_vector(fan::camera* camera, const std::string& path, uint_t block_size);
		//rectangle_vector(fan::camera* camera, const fan::color& color, uint_t block_size);
		~rectangle_vector();

		void push_back(const fan::vec3& src, const fan::vec3& dst, const fan::vec2& texture_id, bool queue = false);

		fan::vec3 get_src(uint_t i) const;
		fan::vec3 get_dst(uint_t i) const;
		fan::vec3 get_size(uint_t i) const;

		void set_position(uint_t i, const fan::vec3& src, const fan::vec3& dst, bool queue = false);
		void set_size(uint_t i, const fan::vec3& size, bool queue = false);

		void draw();

		void set_texture(uint_t i, const fan::vec2& texture_id, bool queue = false);

		void generate_textures(const std::string& path, const fan::vec2& block_size);

		void write_textures();

		void release_queue(bool position, bool size, bool textures);

		square_corners get_corners(uint_t i) const;

		uint_t size() const;

	private:

		unsigned int m_texture_ssbo;
		unsigned int m_texture_id_ssbo;

		fan::vec2i block_size;
		fan::vec2i m_amount_of_textures;

		std::vector<int> m_textures;

	};

	class skybox {
	public:
		skybox(
			fan::camera* camera,
			const std::string& left,
			const std::string& right,
			const std::string& front,
			const std::string back,
			const std::string bottom,
			const std::string& top
		);

		~skybox();

		void draw();

	private:
		unsigned int m_texture_id;
		unsigned int m_skybox_vao, m_skybox_vbo;

		fan::shader m_shader;
		fan::camera* m_camera;
		static constexpr f32_t skybox_vertices[108] = {
			-1.0,  1.0, -1.0,
			-1.0, -1.0, -1.0,
			1.0, -1.0, -1.0,
			1.0, -1.0, -1.0,
			1.0,  1.0, -1.0,
			-1.0,  1.0, -1.0,

			-1.0, -1.0,  1.0,
			-1.0, -1.0, -1.0,
			-1.0,  1.0, -1.0,
			-1.0,  1.0, -1.0,
			-1.0,  1.0,  1.0,
			-1.0, -1.0,  1.0,

			1.0, -1.0, -1.0,
			1.0, -1.0,  1.0,
			1.0,  1.0,  1.0,
			1.0,  1.0,  1.0,
			1.0,  1.0, -1.0,
			1.0, -1.0, -1.0,

			-1.0, -1.0,  1.0,
			-1.0,  1.0,  1.0,
			1.0,  1.0,  1.0,
			1.0,  1.0,  1.0,
			1.0, -1.0,  1.0,
			-1.0, -1.0,  1.0,

			-1.0,  1.0, -1.0,
			1.0,  1.0, -1.0,
			1.0,  1.0,  1.0,
			1.0,  1.0,  1.0,
			-1.0,  1.0,  1.0,
			-1.0,  1.0, -1.0,

			-1.0, -1.0, -1.0,
			-1.0, -1.0,  1.0,
			1.0, -1.0, -1.0,
			1.0, -1.0, -1.0,
			-1.0, -1.0,  1.0,
			1.0, -1.0,  1.0
		};
	};

	struct mesh_vertex {
		fan::vec3 position;
		fan::vec3 normal;
		fan::vec2 texture_coordinates;
	};

	struct mesh_texture {
		unsigned int id;
		std::string type;
		aiString path;
	};

	class model_mesh {
	public:
		std::vector<mesh_vertex> vertices;
		std::vector<unsigned int> indices;
		std::vector<mesh_texture> textures;
		unsigned int vao, vbo, ebo;

		model_mesh(
			const std::vector<mesh_vertex>& vertices,
			const std::vector<unsigned int>& indices,
			const std::vector<mesh_texture>& textures
		);

	private:

		void initialize_mesh();
	};

	class model_loader {
	protected:
		model_loader(const std::string& path, const fan::vec3& size);

		std::vector<model_mesh> meshes;
		std::vector<mesh_texture> textures_loaded;
	private:
		void load_model(const std::string& path, const fan::vec3& size);

		void process_node(aiNode* node, const aiScene* scene, const fan::vec3& size);

		model_mesh process_mesh(aiMesh* mesh, const aiScene* scene, const fan::vec3& size);

		std::vector<mesh_texture> load_material_textures(aiMaterial* mat, aiTextureType type, const std::string& type_name);

		std::string directory;
	};

	class model : public model_loader {
	public:
		model(fan::camera* camera);
		model(fan::camera* camera, const std::string& path, const fan::vec3& position, const fan::vec3& size);

		void draw();

		fan::vec3 get_position();
		void set_position(const fan::vec3& position);

		fan::vec3 get_size();
		void set_size(const fan::vec3& size);

		fan::camera* m_camera;

	private:
		fan::shader m_shader;

		fan::vec3 m_position;
		fan::vec3 m_size;

	};


	fan::vec3 line_triangle_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 3, 3>& triangle);
	fan::vec3 line_plane_intersection(const fan::da_t<f32_t, 2, 3>& line, const fan::da_t<f32_t, 4, 3>& square);

	template<uint_t i>
	inline std::conditional_t<i == -1, std::vector<triangle_vertices_t>, triangle_vertices_t> terrain_generator::get_vertices()
	{
		if constexpr(i == (uint_t)-1) {
			return fan_3d::terrain_generator::m_triangle_vertices;
		}
		else {
			return fan_3d::terrain_generator::m_triangle_vertices[i];
		}
	}

}

namespace fan {

	constexpr int world_size = 150;

	template <typename T>
	constexpr auto grid_direction(const T& src, const T& dst) {
		T vector(src - dst);
		return vector / vector.abs().max();
	}

	template <typename T>
	struct grid_raycast_s {
		T direction, begin;
		std::conditional_t<T::size() == 2, fan::vec2i, fan::vec3i> grid;
	};

	template <typename T>
	constexpr bool grid_raycast_single(grid_raycast_s<T>& caster, f32_t grid_size) {
		T position(caster.begin % grid_size); // mod
		for (uint8_t i = 0; i < T::size(); i++) {
			position[i] = ((caster.direction[i] < 0) ? position[i] : grid_size - position[i]);
			position[i] = fan::abs((!caster.direction[i] ? INFINITY : ((!position[i] ? grid_size : position[i]) / caster.direction[i])));
		}
		caster.grid = (caster.begin += caster.direction * position.min()) / grid_size;
		for (uint8_t i = 0; i < T::size(); i++)
			caster.grid[i] -= ((caster.direction[i] < 0) & (position[i] == position.min()));
		return 1;
	}

	template <typename T, typename map_>
	constexpr T grid_raycast(const T& start, const T& end, const map_& map, f32_t block_size) {
		if (start == end) {
			return start;
		}
		grid_raycast_s<T> raycast = { grid_direction(end, start), start, T() };
		T distance = end - start;
		auto max = distance.abs().max();
		for (uint_t i = 0; i < max; i++) {
			fan::grid_raycast_single<T>(raycast, block_size);
			if constexpr (T::size() == 2) {
				if (raycast.grid[0] < 0 || raycast.grid[1] < 0 ||
				raycast.grid[0] >= world_size || raycast.grid[1] >= world_size) {
				continue;
			}
				if (map[(int)raycast.grid[0]][(int)raycast.grid[1]]) {
					return raycast.grid;
				}
			}
			else {
				if (raycast.grid[0] < 0 || raycast.grid[1] < 0 || raycast.grid[2] < 0 ||
				raycast.grid[0] >= world_size || raycast.grid[1] >= world_size || raycast.grid[2] >= world_size) {
				continue;
			}
				if (map[(int)raycast.grid[0]][(int)raycast.grid[1]][(int)raycast.grid[2]]) {
					return raycast.grid;
				}
			}
		}
		return T(fan::RAY_DID_NOT_HIT);
	}

	#define d_grid_raycast_2d(start, end, raycast, block_size) \
		fan::grid_raycast_s<fan::vec2> raycast = { grid_direction(end, start), start, fan::vec2() }; \
		f_t _private_travel_distance = fan_2d::distance((start / block_size).floored(), (end / block_size).floored()); \
		if (!(start == end)) \
			while(grid_raycast_single(raycast, block_size) && _private_travel_distance >= fan_2d::distance((start / block_size).floored(), raycast.grid))

	#define d_grid_raycast_3d(start, end, raycast, block_size) \
		fan::grid_raycast_s<fan::vec3> raycast = { grid_direction(end, start), start, fan::vec3() }; \
		if (!(start == end)) \
			while(grid_raycast_single(raycast, block_size))

	void depth_test(bool value);

}