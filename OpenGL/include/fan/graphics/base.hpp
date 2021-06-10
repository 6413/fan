#pragma once

#include <fan/types/types.hpp>
#include <fan/types/color.hpp>
#include <fan/graphics/camera.hpp>
#include <fan/graphics/shader.hpp>

namespace fan {

#if SYSTEM_BIT == 32
	constexpr auto GL_FLOAT_T = GL_FLOAT;
#else
	// for now
	constexpr auto GL_FLOAT_T = GL_FLOAT;
#endif

	enum class opengl_buffer_type {
		buffer_object,
		vertex_array_object,
		shader_storage_buffer_object,
		texture,
		frame_buffer_object,
		render_buffer_object,
		last
	};

	static void bind_vao(uint32_t vao, const std::function<void()>& function)
	{
		glBindVertexArray(vao);
		function();
		glBindVertexArray(0);
	}

	static void write_glbuffer(unsigned int buffer, void* data, uint_t size, uint_t target = GL_ARRAY_BUFFER, uint_t location = fan::uninitialized)
	{
		glBindBuffer(target, buffer);
		glBufferData(target, size, data, GL_STATIC_DRAW);
		glBindBuffer(target, 0);
		if (target == GL_SHADER_STORAGE_BUFFER) {
			glBindBufferBase(target, location, buffer);
		}
	}

	static void edit_glbuffer(unsigned int buffer, void* data, uint_t offset, uint_t size, uint_t target = GL_ARRAY_BUFFER, uint_t location = fan::uninitialized)
	{
		glBindBuffer(target, buffer);
		glBufferSubData(target, offset, size, data);
		glBindBuffer(target, 0);
		if (target == GL_SHADER_STORAGE_BUFFER) {
			glBindBufferBase(target, location, buffer);
		}
	}

	template <opengl_buffer_type buffer_type, opengl_buffer_type I = opengl_buffer_type::last, typename ...T>
	constexpr auto comparer(const T&... x) {
		if constexpr (buffer_type == I) {
			std::get<static_cast<int>(I)>(std::forward_as_tuple(x...))();
		}
		else if (static_cast<int>(I) > 0) {
			comparer<buffer_type, static_cast<opengl_buffer_type>(static_cast<int>(I) - 1)>(x...);
		}
	}

	template <uint_t T_layout_location, opengl_buffer_type T_buffer_type, bool gl_3_0_attribute = false, uint32 gl_buffer_type = (uint32_t)-1> 
	class glsl_location_handler { 

	public:

		uint32_t m_buffer_object;

		glsl_location_handler() : m_buffer_object(fan::uninitialized) {
			this->allocate_buffer();
		}

		glsl_location_handler(uint32_t buffer_object) : m_buffer_object(buffer_object) {}

		~glsl_location_handler() {
			this->free_buffer();
		}


		glsl_location_handler(const glsl_location_handler& handler) : m_buffer_object(fan::uninitialized) {
			this->allocate_buffer();
		}

		glsl_location_handler(glsl_location_handler&& handler) : m_buffer_object(fan::uninitialized) {
			if ((int)handler.m_buffer_object == fan::uninitialized) {
				throw std::runtime_error("attempting to move unallocated memory");
			}
			this->operator=(std::move(handler));
		}

		glsl_location_handler& operator=(const glsl_location_handler& handler) {

			this->free_buffer();

			this->allocate_buffer();

			return *this;
		}

		glsl_location_handler& operator=(glsl_location_handler&& handler) {

			this->free_buffer();

			this->m_buffer_object = handler.m_buffer_object;

			handler.m_buffer_object = fan::uninitialized;

			return *this;
		}

	protected:

		static constexpr uint32_t gl_buffer =
			conditional_value<gl_buffer_type != (uint32_t)-1, gl_buffer_type, conditional_value<T_buffer_type == fan::opengl_buffer_type::buffer_object, GL_ARRAY_BUFFER, 
			conditional_value<T_buffer_type == fan::opengl_buffer_type::vertex_array_object, 0,
			conditional_value<T_buffer_type == fan::opengl_buffer_type::texture, GL_TEXTURE_2D, 
			conditional_value<T_buffer_type == fan::opengl_buffer_type::shader_storage_buffer_object, GL_SHADER_STORAGE_BUFFER, 
			conditional_value<T_buffer_type == fan::opengl_buffer_type::frame_buffer_object, GL_FRAMEBUFFER, 
			conditional_value<T_buffer_type == fan::opengl_buffer_type::render_buffer_object, GL_RENDERBUFFER, static_cast<uint32_t>(fan::uninitialized)
			>::value>::value>::value>::value>::value>::value>::value;

		void allocate_buffer() {
			comparer<T_buffer_type>(
				[&] { glGenBuffers(1, &m_buffer_object); },
				[&] { glGenVertexArrays(1, &m_buffer_object); },
				[&] { glGenBuffers(1, &m_buffer_object); },
				[&] { glGenTextures(1, &m_buffer_object); },
				[&] { glGenFramebuffers(1, &m_buffer_object); },
				[&] { glGenRenderbuffers(1, &m_buffer_object); }
			);
		}

		void free_buffer() {

			fan_validate_buffer(m_buffer_object, {
				comparer<T_buffer_type>(
					[&] { glDeleteBuffers(1, &m_buffer_object); },
				[&] { glDeleteVertexArrays(1, &m_buffer_object); },
				[&] { glDeleteBuffers(1, &m_buffer_object); },
				[&] { glDeleteTextures(1, &m_buffer_object); },
				[&] { glDeleteFramebuffers(1, &m_buffer_object); },
				[&] { glDeleteRenderbuffers(1, &m_buffer_object); }
			);
			m_buffer_object = fan::uninitialized;
				});
		}

		void bind_gl_storage_buffer(const std::function<void()> function) const {
			if constexpr (gl_buffer == GL_TEXTURE_2D) {
				glBindTexture(gl_buffer, m_buffer_object);
				function();
				glBindTexture(gl_buffer, 0);
			}
			else if constexpr (gl_buffer == GL_FRAMEBUFFER) {
				glBindFramebuffer(gl_buffer, m_buffer_object);
				function();
				glBindFramebuffer(gl_buffer, 0);
			}
			else if constexpr (gl_buffer == GL_RENDERBUFFER) {
				glBindRenderbuffer(gl_buffer, m_buffer_object);
				function();
				glBindRenderbuffer(gl_buffer, 0);
			}
			else {
				glBindBuffer(gl_buffer, m_buffer_object);
				function();
				glBindBuffer(gl_buffer, 0);
			}
		}

		template <opengl_buffer_type T = T_buffer_type, typename = std::enable_if_t<T == opengl_buffer_type::shader_storage_buffer_object>>
		void bind_gl_storage_buffer_base() const {
			glBindBufferBase(gl_buffer, T_layout_location, m_buffer_object);
		}

		template <opengl_buffer_type T = T_buffer_type, typename = std::enable_if_t<T != opengl_buffer_type::shader_storage_buffer_object>>
		void bind() const {
			glBindBuffer(gl_buffer, m_buffer_object);
		}

		template <opengl_buffer_type T = T_buffer_type, typename = std::enable_if_t<T != opengl_buffer_type::shader_storage_buffer_object>>
		void unbind() const {
			glBindBuffer(gl_buffer, 0);
		}

		template <opengl_buffer_type T = T_buffer_type, typename = std::enable_if_t<T != opengl_buffer_type::texture && T != opengl_buffer_type::vertex_array_object>>
		void edit_data(void* data, uint_t offset, uint_t byte_size) {
			fan::edit_glbuffer(m_buffer_object, data, offset, byte_size, gl_buffer, T_layout_location);
		}

		template <opengl_buffer_type T = T_buffer_type, typename = std::enable_if_t<T == opengl_buffer_type::buffer_object && T != opengl_buffer_type::vertex_array_object>>
		void edit_data(uint_t i, void* data, uint_t byte_size_single) {
			fan::edit_glbuffer(m_buffer_object, data, i * byte_size_single, byte_size_single, gl_buffer, T_layout_location);
		}

		template <bool attribute = gl_3_0_attribute, typename = std::enable_if_t<attribute>>
		void initialize_buffers(void* data, uint_t byte_size, bool divisor, uint_t attrib_count, uint32_t program, const std::string& name) {

			comparer<T_buffer_type>(

				[&] {

				glBindBuffer(gl_buffer, m_buffer_object);

				auto location = glGetAttribLocation(program, name.c_str());

				glEnableVertexAttribArray(location);
				glVertexAttribPointer(location, attrib_count, fan::GL_FLOAT_T, GL_FALSE, 0, 0);

				if (divisor) {
					glVertexAttribDivisor(location, 1);
				}

				this->write_data(data, byte_size);
			},

				[] {}, 

				[&] {
				glBindBuffer(gl_buffer, m_buffer_object); 
				glBindBufferBase(gl_buffer, T_layout_location, m_buffer_object);

				if (divisor) {
					glVertexAttribDivisor(T_layout_location, 1);
				}

				this->write_data(data, byte_size);
			},

				[] {}

			); 
		}

		template <bool attribute = gl_3_0_attribute, typename = std::enable_if_t<!attribute>>
		void initialize_buffers(void* data, uint_t byte_size, bool divisor, uint_t attrib_count) {

			comparer<T_buffer_type>(

				[&] {
				glBindBuffer(gl_buffer, m_buffer_object);

				glEnableVertexAttribArray(T_layout_location);
				glVertexAttribPointer(T_layout_location, attrib_count, fan::GL_FLOAT_T, GL_FALSE, 0, 0);

				if (divisor) {
					glVertexAttribDivisor(T_layout_location, 1);
				}

				this->write_data(data, byte_size);
			},

				[] {}, 

				[&] {
				glBindBuffer(gl_buffer, m_buffer_object); 
				glBindBufferBase(gl_buffer, T_layout_location, m_buffer_object);

				if (divisor) {
					glVertexAttribDivisor(T_layout_location, 1);
				}

				this->write_data(data, byte_size);
			},

				[] {}

			); 
		};


		template <opengl_buffer_type T = T_buffer_type, typename = std::enable_if_t<T == opengl_buffer_type::vertex_array_object>>
		void initialize_buffers(uint32_t vao, const std::function<void()>& binder) {
			glBindVertexArray(vao);
			binder();
			glBindVertexArray(0);
		}

		void write_data(void* data, uint_t byte_size) {
			fan::write_glbuffer(m_buffer_object, data, byte_size, gl_buffer, T_layout_location); 
		}
	};

	template <uint_t _Location = 0>
	class vao_handler : public glsl_location_handler<_Location, fan::opengl_buffer_type::vertex_array_object> {};

	template <uint_t _Location = 0>
	class ebo_handler : public glsl_location_handler<_Location, fan::opengl_buffer_type::buffer_object> {};

	template <uint_t _Location = 0>
	class vbo_handler : public glsl_location_handler<_Location, fan::opengl_buffer_type::buffer_object, true> {};

	template <uint_t _Location = 0>
	class texture_handler : public glsl_location_handler<_Location, fan::opengl_buffer_type::texture> {};

	template <uint_t _Location = 0>
	class render_buffer_handler : public glsl_location_handler<_Location, fan::opengl_buffer_type::render_buffer_object> {};

	template <uint_t _Location = 0>
	class frame_buffer_handler : public glsl_location_handler<_Location, fan::opengl_buffer_type::frame_buffer_object> {};

	#define enable_function_for_vector 	   template<typename T = void, typename = typename std::enable_if<std::is_same<T, T>::value && enable_vector>::type>
	#define enable_function_for_non_vector template<typename T = void, typename = typename std::enable_if<std::is_same<T, T>::value && !enable_vector>::type>
	#define enable_function_for_vector_cpp template<typename T, typename enable_t>

	
	template <typename _Type, uint_t layout_location, bool enable_vector = true, opengl_buffer_type buffer_type = opengl_buffer_type::buffer_object, bool gl_3_0_attrib = true, uint32_t gl_buffer_type = (uint32_t)-1>
	class buffer_object : public glsl_location_handler<layout_location, buffer_type, gl_3_0_attrib, gl_buffer_type> {
	public:

		using value_type = _Type;

		buffer_object() : glsl_location_handler<layout_location, buffer_type, gl_3_0_attrib, gl_buffer_type>() {}

		buffer_object(const _Type& position) : buffer_object()
		{
			if constexpr (enable_vector) {
				this->basic_push_back(position, true);
			}
			else {
				this->m_buffer_object = position;
			}
		}

		buffer_object(const buffer_object& vector) : glsl_location_handler<layout_location, buffer_type, gl_3_0_attrib, gl_buffer_type>(vector) {
			this->m_buffer_object = vector.m_buffer_object;
		}
		buffer_object(buffer_object&& vector) noexcept : glsl_location_handler<layout_location, buffer_type, gl_3_0_attrib, gl_buffer_type>(std::move(vector)) {
			this->m_buffer_object = std::move(vector.m_buffer_object);
		}

		buffer_object& operator=(const buffer_object& vector) {
			buffer_object::glsl_location_handler::operator=(vector);

			this->m_buffer_object = vector.m_buffer_object;

			return *this;
		}
		buffer_object& operator=(buffer_object&& vector) {
			buffer_object::glsl_location_handler::operator=(std::move(vector));

			this->m_buffer_object = std::move(vector.m_buffer_object);

			return *this;
		}

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void reserve(uint_t new_size) {
			m_buffer_object.reserve(new_size);
		}
		enable_function_for_vector void resize(uint_t new_size) {
			m_buffer_object.resize(new_size);
		}
		enable_function_for_vector void resize(uint_t new_size, _Type value) {
			m_buffer_object.resize(new_size, value);
		}

		enable_function_for_vector std::vector<_Type> get_values() const {
			return this->m_buffer_object;
		}

		enable_function_for_vector void set_values(const std::vector<_Type>& values) {
			this->m_buffer_object.clear();
			this->m_buffer_object.insert(this->m_buffer_object.begin(), values.begin(), values.end());
		}

		enable_function_for_vector _Type get_value(uint_t i) const {
			return this->m_buffer_object[i];
		}
		enable_function_for_vector void set_value(uint_t i, const _Type& value, bool queue = false) {
			this->m_buffer_object[i] = value;

			if (!queue) {
				this->edit_data(i);
			}
		}

		enable_function_for_vector void erase(uint_t i, bool queue = false) {
			this->m_buffer_object.erase(this->m_buffer_object.begin() + i);

			if (!queue) {
				this->write_data();
			}
		}
		enable_function_for_vector void erase(uint_t begin, uint_t end, bool queue = false) {
			if (!begin && end == this->m_buffer_object.size()) {
				this->m_buffer_object.clear();
			}
			else {
				this->m_buffer_object.erase(this->m_buffer_object.begin() + begin, this->m_buffer_object.begin() + end);
			}

			if (!queue) {
				this->write_data();
			}
		}

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector _Type get_value() const {
			return m_buffer_object;
		}
		enable_function_for_non_vector void set_value(const _Type& value) {
			this->m_buffer_object = value;
		}

		// -----------------------------------------------------


		enable_function_for_vector void initialize_buffers(bool divisor, uint32_t attrib_count) {
			buffer_object::glsl_location_handler::initialize_buffers(m_buffer_object.data(), sizeof(_Type) * m_buffer_object.size(), divisor, attrib_count);
		}

		enable_function_for_vector void initialize_buffers(uint_t program, const std::string& path, bool divisor, uint32_t attrib_count) {
			//glBindAttribLocation(program, layout_location, path.c_str());
			buffer_object::glsl_location_handler::initialize_buffers(m_buffer_object.data(), sizeof(_Type) * m_buffer_object.size(), divisor, attrib_count, program, path);
		}

		enable_function_for_vector void basic_push_back(const _Type& value, bool queue = false) {
			this->m_buffer_object.emplace_back(value);
			if (!queue) {
				this->write_data();
			}
		}

		enable_function_for_vector void edit_data(uint_t i) {
			buffer_object::glsl_location_handler::edit_data(i, m_buffer_object.data() + i, sizeof(_Type));
		}
		enable_function_for_vector void write_data() {
			buffer_object::glsl_location_handler::write_data(m_buffer_object.data(), sizeof(_Type) * m_buffer_object.size());
		}

		enable_function_for_vector auto size() const {
			return this->m_buffer_object.size();
		}


	protected:

		// ----------------------------------------------------- vector enabled functions


		// -----------------------------------------------------

		std::conditional_t<enable_vector, std::vector<_Type>, _Type> m_buffer_object;

	};


	template <bool enable_vector, typename _Vector, uint_t layout_location = 1, opengl_buffer_type buffer_type = opengl_buffer_type::buffer_object, bool gl_3_0_attrib = false>
	class basic_shape_position : public glsl_location_handler<layout_location, buffer_type, gl_3_0_attrib> {
	public:

		basic_shape_position() : glsl_location_handler<layout_location, buffer_type, gl_3_0_attrib>() {}

		basic_shape_position(const _Vector& position) : basic_shape_position()
		{
			if constexpr (enable_vector) {
				this->basic_push_back(position, true);
			}
			else {
				this->m_position = position;
			}
		}

		basic_shape_position(const basic_shape_position& vector) : glsl_location_handler<layout_location, buffer_type, gl_3_0_attrib>(vector) {
			this->m_position = vector.m_position;
		}
		basic_shape_position(basic_shape_position&& vector) noexcept : glsl_location_handler<layout_location, buffer_type, gl_3_0_attrib>(std::move(vector)) {
			this->m_position = std::move(vector.m_position);
		}

		basic_shape_position& operator=(const basic_shape_position& vector) {
			basic_shape_position::glsl_location_handler::operator=(vector);

			this->m_position = vector.m_position;

			return *this;
		}
		basic_shape_position& operator=(basic_shape_position&& vector) {
			basic_shape_position::glsl_location_handler::operator=(std::move(vector));

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


		enable_function_for_vector void initialize_buffers(bool divisor) {
			basic_shape_position::glsl_location_handler::initialize_buffers(m_position.data(), sizeof(_Vector) * m_position.size(), divisor, _Vector::size());
		}

		enable_function_for_vector void initialize_buffers(uint_t program, const std::string& path, bool divisor) {
			basic_shape_position::glsl_location_handler::initialize_buffers(m_position.data(), sizeof(_Vector) * m_position.size(), divisor, _Vector::size(), program, path);
		}

		enable_function_for_vector void basic_push_back(const _Vector& position, bool queue = false) {
			this->m_position.emplace_back(position);
			if (!queue) {
				this->write_data();
			}
		}

		enable_function_for_vector void edit_data(uint_t i) {
			basic_shape_position::glsl_location_handler::edit_data(i, m_position.data() + i, sizeof(_Vector));
		}
		enable_function_for_vector void write_data() {
			basic_shape_position::glsl_location_handler::write_data(m_position.data(), sizeof(_Vector) * m_position.size());
		}


	protected:

		// ----------------------------------------------------- vector enabled functions


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

	template <bool enable_vector, uint_t layout_location = 0, opengl_buffer_type buffer_type = opengl_buffer_type::buffer_object, bool gl_3_0_attrib = false>
	class basic_shape_color_vector : public glsl_location_handler<layout_location, buffer_type, gl_3_0_attrib> {
	public:

		basic_shape_color_vector() : basic_shape_color_vector::glsl_location_handler() {}
		basic_shape_color_vector(const fan::color& color) : basic_shape_color_vector() {
			if constexpr (enable_vector) {
				basic_push_back(color, true);
			}
			else {
				this->m_color = color;
			}
		}
		basic_shape_color_vector(const basic_shape_color_vector& vector) : basic_shape_color_vector::glsl_location_handler(vector) {
			this->m_color = vector.m_color;
		}
		basic_shape_color_vector(basic_shape_color_vector&& vector) noexcept : basic_shape_color_vector::glsl_location_handler(std::move(vector)) {
			this->m_color = std::move(vector.m_color);
		}

		basic_shape_color_vector& operator=(const basic_shape_color_vector& vector) {
			basic_shape_color_vector::glsl_location_handler::operator=(vector);

			this->m_color = vector.m_color;

			return *this;
		}
		basic_shape_color_vector& operator=(basic_shape_color_vector&& vector) noexcept {
			basic_shape_color_vector::glsl_location_handler::operator=(std::move(vector));

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

		enable_function_for_vector fan::color get_color(uint_t i) const {
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
			basic_shape_color_vector::glsl_location_handler::edit_data(i, m_color.data() + i, sizeof(fan::color));
		}
		enable_function_for_vector void edit_data(void* data, uint_t offset, uint_t byte_size) {
			basic_shape_color_vector::glsl_location_handler::edit_data(data, offset, byte_size);
		}

		enable_function_for_vector void write_data() {
			basic_shape_color_vector::glsl_location_handler::write_data(m_color.data(), sizeof(fan::color) * m_color.size());
		}

		template<typename T = void, 
			bool enable_function_t = gl_3_0_attrib, 
			typename = typename 
			std::enable_if<std::is_same<T, T>::value && 
			enable_vector && !enable_function_t>::type 

		> void initialize_buffers(bool divisor) {
			basic_shape_color_vector::glsl_location_handler::initialize_buffers(m_color.data(), sizeof(fan::color) * m_color.size(), divisor, fan::color::size());
		}

		template<typename T = void, 
			bool enable_function_t = gl_3_0_attrib, 
			typename = typename
			std::enable_if<std::is_same<T, T>::value && 
			enable_vector && enable_function_t>::type
		> 
			void initialize_buffers(uint_t program, const std::string& path, bool divisor) {
			basic_shape_color_vector::glsl_location_handler::initialize_buffers(m_color.data(), sizeof(fan::color) * m_color.size(), divisor, fan::color::size(), program, path);
		}
		// -----------------------------------------------------

		std::conditional_t<enable_vector, std::vector<fan::color>, fan::color> m_color;

	};

	template <bool enable_vector, typename _Vector>
	class basic_shape : 
		public basic_shape_position<enable_vector, _Vector>, 
		public basic_shape_size<enable_vector, _Vector>,
		public vao_handler<> {
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
			glBindVertexArray(vao_handler::m_buffer_object);
			if (i != (uint_t)fan::uninitialized) {
				glDrawArraysInstancedBaseInstance(mode, 0, count, 1, i);
			}
			else {
				glDrawArrays(mode, 0, primcount * count);
			}

			glBindVertexArray(0);
		}

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector void basic_draw(unsigned int mode, uint_t count) const {
			glBindVertexArray(vao_handler::m_buffer_object);
			glDrawArrays(mode, 0, count);
			glBindVertexArray(0);
		}

		// -----------------------------------------------------

		fan::shader m_shader;

	};

	template <typename _Vector>
	class basic_vertice_vector : 
		public basic_shape_position<true, _Vector, 1, fan::opengl_buffer_type::buffer_object, true>, 
		public basic_shape_color_vector<true, 0, fan::opengl_buffer_type::buffer_object, true>,
		public vao_handler<> {
	public:

		basic_vertice_vector(fan::camera* camera, const fan::shader& shader) : m_camera(camera), m_shader(shader) {}
		basic_vertice_vector(fan::camera* camera, const fan::shader& shader, const fan::vec2& position, const fan::color& color) 
			: basic_vertice_vector::basic_shape_position(position), basic_vertice_vector::basic_shape_color_vector(color), 
			  m_camera(camera), m_shader(shader) { }

		basic_vertice_vector(const basic_vertice_vector& vector)	
			: basic_vertice_vector::basic_shape_position(vector), 
			  basic_vertice_vector::basic_shape_color_vector(vector),
			  vao_handler(vector), m_camera(vector.m_camera), m_shader(vector.m_shader) { }

		basic_vertice_vector(basic_vertice_vector&& vector) noexcept 
			: basic_vertice_vector::basic_shape_position(std::move(vector)), 
			  basic_vertice_vector::basic_shape_color_vector(std::move(vector)),
			  vao_handler(std::move(vector)), m_camera(vector.m_camera), 
			  m_shader(std::move(vector.m_shader)) { }

		basic_vertice_vector& operator=(const basic_vertice_vector& vector) {
			basic_vertice_vector::basic_shape_position::operator=(vector);
			basic_vertice_vector::basic_shape_color_vector::operator=(vector);
			vao_handler::operator=(vector);

			m_shader = vector.m_shader;
			m_camera = vector.m_camera;

			return *this;
		}
		basic_vertice_vector& operator=(basic_vertice_vector&& vector) {
			basic_vertice_vector::basic_shape_position::operator=(std::move(vector));
			basic_vertice_vector::basic_shape_color_vector::operator=(std::move(vector));
			vao_handler::operator=(std::move(vector));

			m_shader = std::move(vector.m_shader);
			m_camera = std::move(vector.m_camera);

			return *this;
		}

		void reserve(uint_t size) {
			basic_vertice_vector::basic_shape_position::reserve(size);
			basic_vertice_vector::basic_shape_color_vector::reserve(size);
		}
		void resize(uint_t size, const fan::color& color) {
			basic_vertice_vector::basic_shape_position::resize(size);
			basic_vertice_vector::basic_shape_color_vector::resize(size, color);
		}

		uint_t size() const {
			return basic_vertice_vector::basic_shape_position::m_position.size();
		}

		void erase(uint_t i, bool queue = false) {
			basic_vertice_vector::basic_shape_position::erase(i, queue);
			basic_vertice_vector::basic_shape_color_vector::erase(i, queue);
		}
		void erase(uint_t begin, uint_t end, bool queue = false) {
			basic_vertice_vector::basic_shape_position::erase(begin, end, queue);
			basic_vertice_vector::basic_shape_color_vector::erase(begin, end, queue);
		}

		fan::camera* m_camera;

	protected:

		void basic_push_back(const _Vector& position, const fan::color& color, bool queue = false) {
			basic_vertice_vector::basic_shape_position::basic_push_back(position, queue);
			basic_vertice_vector::basic_shape_color_vector::basic_push_back(color, queue);
		}

		void edit_data(uint_t i, bool position, bool color) {
			if (position) {
				basic_vertice_vector::basic_shape_position::edit_data(i);
			}
			if (color) {
				basic_vertice_vector::basic_shape_color_vector::edit_data(i);
			}
		}

		void write_data(bool position, bool color) {
			if (position) {
				basic_vertice_vector::basic_shape_position::write_data();
			}
			if (color) {
				basic_vertice_vector::basic_shape_color_vector::write_data();
			} 
		}

		void basic_draw(uint_t begin, uint_t end, const std::vector<uint32_t>& indices, unsigned int mode, uint_t count, uint32_t index_restart, uint32_t single_draw_amount) const {
			fan::bind_vao(vao_handler::m_buffer_object, [&] {
				if (begin != (uint32_t)fan::uninitialized && end != (uint32_t)fan::uninitialized) {
					glDrawElements(mode, end == (uint32_t)fan::uninitialized ? single_draw_amount : single_draw_amount * (end - begin) + single_draw_amount, GL_UNSIGNED_INT, (void*)((sizeof(GLuint) * single_draw_amount) * begin));
				}
				else if (begin != (uint32_t)fan::uninitialized && end == (uint32_t)fan::uninitialized) {
					glDrawElements(mode, single_draw_amount, GL_UNSIGNED_INT, (void*)((sizeof(GLuint) * single_draw_amount) * begin));
				}
				else if (begin == (uint32_t)fan::uninitialized && end != (uint32_t)fan::uninitialized) {
					glDrawElements(mode, single_draw_amount * end, GL_UNSIGNED_INT, 0);
				}
				else if (index_restart == UINT32_MAX) {
					glDrawElements(mode, count, GL_UNSIGNED_INT, 0);
				}
				else {
					for (uint32_t j = 0; j < indices.size() / index_restart; j++) {
						glDrawElements(mode, index_restart, GL_UNSIGNED_INT, (void*)((sizeof(GLuint) * index_restart) * j));
					}
				}
			});
		}

		void set_shader(const fan::shader& shader) {
			m_shader = shader;
		}

		fan::shader m_shader;

	};

	template <uint_t layout_location = 0, fan::opengl_buffer_type gl_buffer = fan::opengl_buffer_type::buffer_object, bool gl_3_0_attribute = false>
	class basic_shape_color_vector_vector : public glsl_location_handler<layout_location, gl_buffer, gl_3_0_attribute> {
	public:

		basic_shape_color_vector_vector() {}
		basic_shape_color_vector_vector(const std::vector<fan::color>& color) : basic_shape_color_vector_vector() {
			this->m_color.emplace_back(color);

			this->write_data();
		}

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

		std::vector<fan::color> get_color(uint_t i) {
			return this->m_color[i];
		}
		void set_color(uint_t i, const std::vector<fan::color>& color, bool queue = false) {
			this->m_color[i] = color;
			if (!queue) {
				write_data();
			}
		}

		void erase(uint_t i, bool queue = false) {
			this->m_color.erase(this->m_color.begin() + i);
			if (!queue) {
				this->write_data();
			}
		}


	protected:

		void basic_push_back(const std::vector<fan::color>& color, bool queue = false) {
			this->m_color.emplace_back(color);
			if (!queue) {
				write_data();
			}
		}

		void edit_data(uint_t i) {

			std::vector<fan::color> vector;

			for (int j = 0; j < this->m_color[i].size(); j++) {
				for (int k = 0; k < 6; k++) {
					vector.emplace_back(this->m_color[i][j]);
				}
			}
			
			uint_t offset = 0;

			for (uint_t j = 0; j < i; j++) {
				offset += m_color[j].size();
			}

			// AAAAAAAAA GL_ARRAY_BUFFER
			fan::edit_glbuffer(basic_shape_color_vector_vector::glsl_location_handler::m_buffer_object, vector.data(), sizeof(fan::color) * offset, sizeof(fan::color) * vector.size(), GL_ARRAY_BUFFER, layout_location);
		}

		//void edit_data(uint_t i) {
		//	std::vector<fan::color> vector(this->m_color[i].size());
		//	std::copy(this->m_color[i].begin(), this->m_color[i].end(), vector.begin());
		//	uint_t offset = 0;
		//	for (uint_t j = 0; j < i; j++) {
		//		offset += m_color[j].size();
		//	}

		//	// AAAAAAAAA GL_ARRAY_BUFFER
		//	fan::edit_glbuffer(basic_shape_color_vector_vector::glsl_location_handler::m_buffer_object, vector.data(), sizeof(fan::color) * offset, sizeof(fan::color) * vector.size(), GL_ARRAY_BUFFER, layout_location);
		//}


		void write_data()
		{
			std::vector<fan::color> vector;

			for (uint_t i = 0; i < m_color.size(); i++) {
				for (int j = 0; j < m_color[i].size(); j++) {
					for (int k = 0; k < 6; k++) {
						vector.emplace_back(m_color[i][j]);
					}
				}
			}

			//AAAAAAAAAA GL_ARRAY_BUFFER
			fan::write_glbuffer(basic_shape_color_vector_vector::glsl_location_handler::m_buffer_object, vector.data(), sizeof(fan::color) * vector.size(), GL_ARRAY_BUFFER, layout_location);
		}

		void initialize_buffers(uint_t program, const std::string& name, bool divisor) {
			// AAAAAA
			glBindBuffer(GL_ARRAY_BUFFER, basic_shape_color_vector_vector::glsl_location_handler::m_buffer_object);
			glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);
			if constexpr (gl_buffer == fan::opengl_buffer_type::shader_storage_buffer_object) {
				glBindBufferBase(GL_SHADER_STORAGE_BUFFER, layout_location, basic_shape_color_vector_vector::m_buffer_object);
			}
			else {
				GLint location = glGetAttribLocation(program, name.c_str());
				glEnableVertexAttribArray(location);
				glVertexAttribPointer(location, 4, fan::GL_FLOAT_T, GL_FALSE, sizeof(fan::color), 0);
			}
			if (divisor) {
				glVertexAttribDivisor(0, 1);
			}
		}

		std::vector<std::vector<fan::color>> m_color;

	};
}