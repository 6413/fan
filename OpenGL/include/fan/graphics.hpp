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
		camera(fan::window& window);

		fan::camera& operator=(const fan::camera& camera);
		fan::camera& operator=(fan::camera&& camera) noexcept;

		void move(f32_t movement_speed, bool noclip = true, f32_t friction = 12);
		void rotate_camera(bool when);

		fan::mat4 get_view_matrix() const;
		fan::mat4 get_view_matrix(fan::mat4 m) const;

		fan::vec3 get_position() const;
		void set_position(const fan::vec3& position);

		fan::vec3 get_velocity() const;
		void set_velocity(const fan::vec3& velocity);

		f32_t get_yaw() const;
		void set_yaw(f32_t angle);

		f32_t get_pitch() const;
		void set_pitch(f32_t angle);

		bool first_movement = true;

		void update_view();

		static constexpr f32_t sensitivity = 0.05f;

		static constexpr f32_t max_yaw = 180;
		static constexpr f32_t max_pitch = 89;

		static constexpr f32_t gravity = 500;
		static constexpr f32_t jump_force = 100;

		static constexpr fan::vec3 world_up = fan::vec3(0, 1, 0);

		fan::window& m_window;

	private:

		fan::vec3 m_position;
		fan::vec3 m_front;

		f32_t m_yaw;
		f32_t m_pitch;
		fan::vec3 m_right;
		fan::vec3 m_up;
		fan::vec3 m_velocity;

	};

	static fan::vec2 random_vector(f_t min, f_t max) {
		return fan::vec2(fan::random<int64_t, int64_t>(min, max), fan::random<int64_t, int64_t>(min, max));
	}

	void bind_vao(uint32_t vao, const std::function<void()>& function);

	void write_glbuffer(unsigned int buffer, void* data, uint_t size, uint_t target = GL_ARRAY_BUFFER, uint_t location = fan::uninitialized);
	void edit_glbuffer(unsigned int buffer, void* data, uint_t offset, uint_t size, uint_t target = GL_ARRAY_BUFFER,  uint_t location = fan::uninitialized);

	/*class vao_handler {
	public:

		vao_handler();

		vao_handler(const vao_handler& handler);
		vao_handler(vao_handler&& handler) noexcept;

		vao_handler& operator=(const vao_handler& handler);
		vao_handler& operator=(vao_handler&& handler) noexcept;

		~vao_handler();

		void generate_vertex_array();
		void erase_vertex_array();

	protected:

		uint32_t m_vao;

	};*/

	template <bool _Test, uint_t _Ty1, uint_t _Ty2>
	struct conditional_value {
		static constexpr auto value = _Ty1;
	};

	template <uint_t _Ty1, uint_t _Ty2>
	struct conditional_value<false, _Ty1, _Ty2> {
		static constexpr auto value = _Ty2;
	};

	template <bool _Test, uint_t _Ty1, uint_t _Ty2>
	struct conditional_value_t {
		static constexpr auto value = conditional_value<_Test, _Ty1, _Ty2>::value;
	};

	enum class opengl_buffer_type {
		vbo,
		vao,
		ssbo,
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

	// locations are for vbo, ssbo, make sure locations dont collide

	#define base_object_builder(class_name, variable_name, buffer_type, layout_location) \
	class class_name { \
	public:\
	\
		static constexpr uint_t gl_buffer =\
			conditional_value<buffer_type == opengl_buffer_type::vbo, GL_ARRAY_BUFFER, \
			conditional_value<buffer_type == opengl_buffer_type::vao, 0,\
			conditional_value<buffer_type == opengl_buffer_type::ssbo, GL_SHADER_STORAGE_BUFFER, static_cast<uint_t>(fan::uninitialized)>::value>::value>::value;\
	\
		void allocate_buffer() {\
			comparer<buffer_type>(\
				[&] { glGenBuffers(1, &variable_name); },\
				[&] { glGenVertexArrays(1, &variable_name); },\
				[&] { glGenBuffers(1, &variable_name); },\
				[&] { glGenTextures(1, &variable_name); }\
			);\
		}\
	\
		void free_buffer() {\
			fan_validate_buffer(variable_name, {\
				comparer<buffer_type>(\
					[&] { glDeleteBuffers(1, &variable_name); },\
					[&] { glDeleteVertexArrays(1, &variable_name); },\
					[&] { glDeleteBuffers(1, &variable_name); },\
					[&] { glDeleteTextures(1, &variable_name); }\
				);\
				variable_name = fan::uninitialized;\
			});\
		}\
	\
		class_name() : variable_name(fan::uninitialized) {\
			this->allocate_buffer();\
		}\
	\
		~class_name() {\
			this->free_buffer();\
		}\
	\
		\
		class_name(const class_name& handler) : variable_name(fan::uninitialized) {\
			this->allocate_buffer();\
		}\
	\
		class_name(class_name&& handler) noexcept : variable_name(fan::uninitialized) {\
			this->operator=(std::move(handler));\
		}\
	\
		class_name& operator=(const class_name& handler) {\
	\
			this->free_buffer();\
	\
			this->allocate_buffer();\
	\
			return *this;\
		}\
	\
		class_name& operator=(class_name&& handler) {\
	\
			this->free_buffer();\
	\
			this->variable_name = handler.variable_name;\
	\
			handler.variable_name = fan::uninitialized;\
	\
			return *this;\
		}\
	\
		template <opengl_buffer_type T = buffer_type, typename = std::enable_if_t<T == opengl_buffer_type::ssbo>>\
		void bind_gl_storage_buffer() const {\
			glBindBufferBase(gl_buffer, layout_location, variable_name);\
		}\
	\
		template <opengl_buffer_type T = buffer_type, typename = std::enable_if_t<T != opengl_buffer_type::texture && T != opengl_buffer_type::vao>>\
		void edit_data(void* data, uint_t offset, uint_t byte_size) {\
			fan::edit_glbuffer(variable_name, data, offset, byte_size, gl_buffer, layout_location);\
		}\
	\
		uint32_t variable_name;\
	\
	protected:\
	\
		template <opengl_buffer_type T = buffer_type, typename = std::enable_if_t<T == opengl_buffer_type::vbo && T != opengl_buffer_type::vao>>\
		void edit_data(uint_t i, void* data, uint_t byte_size_single) {\
			fan::edit_glbuffer(variable_name, data, i * byte_size_single, byte_size_single, gl_buffer, layout_location);\
		}\
	\
		template <opengl_buffer_type T = buffer_type, typename = std::enable_if_t<T != opengl_buffer_type::texture && T != opengl_buffer_type::vao>>\
		void initialize_buffers(void* data, uint_t byte_size, bool divisor, uint_t attrib_count) {\
	\
			comparer<buffer_type>(\
	\
				[&] {\
					glBindBuffer(gl_buffer, variable_name); \
				\
					glEnableVertexAttribArray(layout_location);\
					glVertexAttribPointer(layout_location, attrib_count, fan::GL_FLOAT_T, GL_FALSE, 0, 0);\
				\
					if (divisor) {\
						glVertexAttribDivisor(layout_location, 1);\
					}\
				\
					this->write_data(data, byte_size);\
				},\
	\
				[] {}, \
	\
				[&] {\
					glBindBuffer(gl_buffer, variable_name); \
					glBindBufferBase(gl_buffer, layout_location, variable_name);\
				\
					if (divisor) {\
						glVertexAttribDivisor(layout_location, 1);\
					}\
				\
					this->write_data(data, byte_size);\
				},\
	\
				[] {}\
				\
				); \
		}\
		\
		template <opengl_buffer_type T = buffer_type, typename = std::enable_if_t<T == opengl_buffer_type::vao>>\
		void initialize_buffers(uint32_t vao, const std::function<void()>&binder) {\
		\
			binder(); \
	}\
	\
	void write_data(void* data, uint_t byte_size) {\
			\
			fan::write_glbuffer(variable_name, data, byte_size, gl_buffer, layout_location); \
		}\
	\
	};

	base_object_builder(vao_handler, m_vao, opengl_buffer_type::vao, 0);

	base_object_builder(texture_handler, m_texture, opengl_buffer_type::texture, 0);

	template <uint_t layout_location, opengl_buffer_type buffer_type>
	base_object_builder(glsl_location_handler, m_buffer_object, buffer_type, layout_location);

	template <typename _Vector, uint_t layout_location = 1, opengl_buffer_type buffer_type = opengl_buffer_type::vbo>
	class basic_shape_position_vector : public glsl_location_handler<layout_location, buffer_type> {
	public:

		basic_shape_position_vector();
		// initialize will write the data not constructor
		basic_shape_position_vector(const _Vector& position);

		basic_shape_position_vector(const basic_shape_position_vector& vector);
		basic_shape_position_vector(basic_shape_position_vector&& vector) noexcept;

		basic_shape_position_vector& operator=(const basic_shape_position_vector& vector);
		basic_shape_position_vector& operator=(basic_shape_position_vector&& vector);

		void resize(uint_t size) {
			m_position.resize(size);
		}

		std::vector<_Vector> get_positions() const {
			return this->m_position;
		}
		void set_positions(const std::vector<_Vector>& positions);

		virtual _Vector get_position(uint_t i) const;
		void set_position(uint_t i, const _Vector& position, bool queue = false);

		void erase(uint_t i, bool queue = false);
		virtual void erase(uint_t begin, uint_t end, bool queue = false);

	protected:

		void initialize_buffers(bool divisor);

		void basic_push_back(const _Vector& position, bool queue = false);

		void edit_data(uint_t i);
		void write_data();

		std::vector<_Vector> m_position;

	};

	template <typename _Vector, uint_t layout_location = 2, opengl_buffer_type buffer_type = opengl_buffer_type::vbo>
	class basic_shape_size_vector : public glsl_location_handler<layout_location, buffer_type> {
	public:

		basic_shape_size_vector();
		basic_shape_size_vector(const _Vector& size);

		basic_shape_size_vector(const basic_shape_size_vector& vector);
		basic_shape_size_vector(basic_shape_size_vector&& vector) noexcept;

		basic_shape_size_vector& operator=(const basic_shape_size_vector& vector);
		basic_shape_size_vector& operator=(basic_shape_size_vector&& vector) noexcept;

		void resize(uint_t size) {
			m_size.resize(size);
		}

		std::vector<_Vector> get_sizes() const {
			return m_size;
		}

		virtual _Vector get_size(uint_t i) const;
		void set_size(uint_t i, const _Vector& size, bool queue = false);

		void erase(uint_t i, bool queue = false);
		void erase(uint_t begin, uint_t end, bool queue = false);

	protected:

		void basic_push_back(const _Vector& size, bool queue = false);

		void edit_data(uint_t i);
		void write_data();

		void initialize_buffers(bool divisor);

		std::vector<_Vector> m_size;

	};

	template <uint_t layout_location = 0, opengl_buffer_type buffer_type = opengl_buffer_type::vbo>
	class basic_shape_color_vector : public glsl_location_handler<layout_location, buffer_type> {
	public:

		basic_shape_color_vector();
		basic_shape_color_vector(const basic_shape_color_vector& vector);
		basic_shape_color_vector(basic_shape_color_vector&& vector) noexcept;

		basic_shape_color_vector& operator=(const basic_shape_color_vector& vector);
		basic_shape_color_vector& operator=(basic_shape_color_vector&& vector) noexcept;

		basic_shape_color_vector(const fan::color& color) : basic_shape_color_vector() {
			basic_push_back(color, true);
		}

		void resize(uint_t size, const fan::color& color) {
			m_color.resize(size, color);
		}

		void resize(uint_t size) {
			m_color.resize(size);
		}

		fan::color get_color(uint_t i)
		{
			return this->m_color[i];
		}

		void set_color(uint_t i, const fan::color& color, bool queue = false) 
		{
			this->m_color[i] = color;
			if (!queue) {
				this->edit_data(i);
			}
		}

		void erase(uint_t i, bool queue = false);
		virtual void erase(uint_t begin, uint_t end, bool queue = false);

	protected:

		void basic_push_back(const fan::color& color, bool queue = false);
	
		void edit_data(uint_t i);

		void write_data();

		void initialize_buffers(bool divisor);

		std::vector<fan::color> m_color;

	};

	template <typename _Vector>
	class basic_shape_velocity_vector {
	public:

		basic_shape_velocity_vector();
		basic_shape_velocity_vector(const _Vector& velocity);

		basic_shape_velocity_vector(const basic_shape_velocity_vector& vector);
		basic_shape_velocity_vector(basic_shape_velocity_vector&& vector) noexcept;

		basic_shape_velocity_vector& operator=(const basic_shape_velocity_vector& vector);
		basic_shape_velocity_vector& operator=(basic_shape_velocity_vector&& vector) noexcept;

		_Vector get_velocity(uint_t i) const {
			return this->m_velocity[i];
		}

		void set_velocity(uint_t i, const _Vector& velocity) {
			this->m_velocity[i] = velocity;
		}

	protected:

		std::vector<_Vector> m_velocity;

	};

	template <typename _Vector>
	class basic_shape_vector : 
		public basic_shape_position_vector<_Vector>, 
		public basic_shape_size_vector<_Vector>,
		public vao_handler {
	public:
		
		basic_shape_vector(fan::camera& camera);
		basic_shape_vector(fan::camera& camera, const fan::shader& shader);
		basic_shape_vector(fan::camera& camera, const fan::shader& shader, const _Vector& position, const _Vector& size);

		basic_shape_vector(const basic_shape_vector& vector);
		basic_shape_vector(basic_shape_vector&& vector) noexcept;

		basic_shape_vector& operator=(const basic_shape_vector& vector);
		basic_shape_vector& operator=(basic_shape_vector&& vector) noexcept;

		void resize(uint_t size);

		virtual uint_t size() const;

	protected:

		void basic_push_back(const _Vector& position, const _Vector& size, bool queue = false);

		void erase(uint_t i, bool queue = false);
		void erase(uint_t begin, uint_t end, bool queue = false);

		void edit_data(uint_t i, bool position, bool size);

		void write_data(bool position, bool size);

		void basic_draw(unsigned int mode, uint_t count, uint_t primcount, uint_t i = fan::uninitialized);

		fan::shader m_shader;

		fan::camera& m_camera;

		fan::window& m_window;

	};

	template <typename _Vector>
	class basic_vertice_vector : 
		public basic_shape_position_vector<_Vector>, 
		public basic_shape_color_vector<>, 
		public basic_shape_velocity_vector<_Vector>,
	    public vao_handler {
	public:

		basic_vertice_vector(fan::camera& camera, const fan::shader& shader);
		basic_vertice_vector(fan::camera& camera, const fan::shader& shader, const fan::vec2& position, const fan::color& color);

		basic_vertice_vector(const basic_vertice_vector& vector) ;
		basic_vertice_vector(basic_vertice_vector&& vector) noexcept;

		~basic_vertice_vector();

		basic_vertice_vector& operator=(const basic_vertice_vector& vector);
		basic_vertice_vector& operator=(basic_vertice_vector&& vector);

		uint_t size() const;

		void erase(uint_t i, bool queue = false);
		void erase(uint_t begin, uint_t end, bool queue = false);

	protected:

		void basic_push_back(const _Vector& position, const fan::color& color, bool queue = false);

		void edit_data(uint_t i, bool position, bool color);

		void write_data(bool position, bool color);

		void basic_draw(unsigned int mode, uint_t count);

		fan::shader m_shader;

		fan::window& m_window;
		fan::camera& m_camera;

	};

	template <uint_t layout_location = 0, uint_t gl_buffer = GL_ARRAY_BUFFER>
	class basic_shape_color_vector_vector {
	public:

		basic_shape_color_vector_vector();
		basic_shape_color_vector_vector(const std::vector<fan::color>& color);
		~basic_shape_color_vector_vector();

		std::vector<fan::color> get_color(uint_t i);
		void set_color(uint_t i, const std::vector<fan::color>& color, bool queue = false);

		void erase(uint_t i, bool queue = false);

		template <typename = std::enable_if_t<gl_buffer == GL_SHADER_STORAGE_BUFFER>>
		void bind_gl_storage_buffer() const {
			glBindBufferBase(gl_buffer, layout_location, m_color_vbo);
		}

	protected:

		void basic_push_back(const std::vector<fan::color>& color, bool queue = false);
	
		void edit_data(uint_t i);
		void edit_data(void* data, uint_t offset, uint_t size);

		void write_data()
		{
			std::vector<fan::color> vector;

			for (uint_t i = 0; i < m_color.size(); i++) {
				vector.insert(vector.end(), m_color[i].begin(), m_color[i].end());
			}

			fan::write_glbuffer(m_color_vbo, vector.data(), sizeof(fan::color) * vector.size(), gl_buffer, layout_location);
		}

		void initialize_buffers(bool divisor = true);

		uint32_t m_color_vbo;

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
		constexpr auto text_renderer_vs("include/fan/glsl/2D/text.vs");
		constexpr auto text_renderer_fs("include/fan/glsl/2D/text.fs");

		constexpr auto single_shapes_vs("include/fan/glsl/2D/shapes.vs");
		constexpr auto single_shapes_fs("include/fan/glsl/2D/shapes.fs");

		constexpr auto single_shapes_bloom_vs("include/fan/glsl/2D/bloom.vs");
		constexpr auto single_shapes_bloom_fs("include/fan/glsl/2D/bloom.fs");
		constexpr auto single_shapes_blur_vs("include/fan/glsl/2D/blur.vs");
		constexpr auto single_shapes_blur_fs("include/fan/glsl/2D/blur.fs");

		constexpr auto single_shapes_bloom_final_vs("include/fan/glsl/2D/bloom_final.vs");
		constexpr auto single_shapes_bloom_final_fs("include/fan/glsl/2D/bloom_final.fs");

		constexpr auto single_sprite_vs("include/fan/glsl/2D/sprite.vs");
		constexpr auto single_sprite_fs("include/fan/glsl/2D/sprite.fs");

		constexpr auto shape_vector_vs("include/fan/glsl/2D/shape_vector.vs");
		constexpr auto shape_vector_fs("include/fan/glsl/2D/shapes.fs");
		constexpr auto sprite_vector_vs("include/fan/glsl/2D/sprite_vector.vs");
		constexpr auto sprite_vector_fs("include/fan/glsl/2D/sprite_vector.fs");
	}

	// returns how much object moved
	fan::vec2 move_object(fan::window& window, fan::vec2& position, fan::vec2& velocity, f32_t speed, f32_t gravity, f32_t jump_force = -800, f32_t friction = 10);

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

	class basic_single_shape {
	public:

		basic_single_shape(fan::camera& camera);
		basic_single_shape(fan::camera& camera, const fan::shader& shader, const fan::vec2& position, const fan::vec2& size);

		~basic_single_shape();

		fan::vec2 get_position() const;
		fan::vec2 get_size() const;
		fan::vec2 get_velocity() const;
		fan::vec2 get_center() const;

		void set_size(const fan::vec2& size);
		virtual void set_position(const fan::vec2& position);
		void set_velocity(const fan::vec2& velocity);

		void basic_draw(GLenum mode, GLsizei count) const;

		virtual fan::vec2 move(f32_t speed, f32_t gravity = 0, f32_t jump_force = -800, f32_t friction = 10);

		bool inside() const;

	//protected:
		fan::vec2 m_position;
		fan::vec2 m_size;

		fan::vec2 m_velocity;

		fan::shader m_shader;

		unsigned int m_vao;

		fan::window& m_window;
		fan::camera& m_camera;
	};

	struct basic_single_color {

		basic_single_color();
		basic_single_color(const fan::color& color);

		fan::color get_color() const;
		void set_color(const fan::color& color);

		fan::color color;

	};

	struct line : protected basic_single_shape, public basic_single_color {

		line(fan::camera& camera);
		line(fan::camera& camera, const fan::mat2& begin_end, const fan::color& color);

		void draw();

		fan::mat2 get_position() const;
		void set_position(const fan::mat2& begin_end);

	};

	struct rectangle : public basic_single_shape, basic_single_color {
		rectangle(fan::camera& camera);
		rectangle(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color);

		rectangle_corners_t get_corners() const;

		fan::vec2 get_center() const;

		f_t get_rotation() const;
		void set_rotation(f_t angle);

		void draw() const;

		fan::vec2 move(f32_t speed, f32_t gravity = 0, f32_t jump_force = -800, f32_t friction = 10) override;

		void set_position(const fan::vec2& position) override;

	private:

		rectangle_corners_t m_corners;

		f_t m_rotation;

	};

	struct image_info {
		fan::vec2i image_size;
		uint32_t texture_id;
	};

	static image_info load_image(const std::string& path, bool flip_image = false);
	static image_info load_image(unsigned char* pixels, const fan::vec2i& size);

	class sprite : public basic_single_shape {
	public:
		sprite(fan::camera& camera);

		// size with default is the size of the image
		sprite(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0, f_t transparency = 1);
		sprite(fan::camera& camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size = 0, f_t transparency = 1);
		sprite(const fan_2d::sprite& sprite);
		sprite(fan_2d::sprite&& sprite);

		void load_sprite(const std::string& path, const fan::vec2i& size = 0, bool flip_image = false);

		void reload_sprite(unsigned char* pixels, const fan::vec2i& size);
		void reload_sprite(const std::string& path, const fan::vec2i& size, bool flip_image = false);

		void draw();

		f32_t get_rotation();
		void set_rotation(f32_t degrees);

	private:

		f32_t m_rotation;
		f_t m_transparency;

		unsigned int m_texture;
	};

	/*class animation : public basic_single_shape {
	public:

		animation(const fan::vec2& position, const fan::vec2& size);

		void add(const std::string& path);

		void draw(uint_t m_texture);

	private:
		std::vector<unsigned int> m_textures;
	};*/

	class vertice_vector : public fan::basic_vertice_vector<fan::vec2> {
	public:

		vertice_vector(fan::camera& camera, uint_t index_restart = UINT32_MAX);
		vertice_vector(fan::camera& camera, const fan::vec2& position, const fan::color& color, uint_t index_restart);
		vertice_vector(const vertice_vector& vector);
		vertice_vector(vertice_vector&& vector) noexcept;
		~vertice_vector();

		vertice_vector& operator=(const vertice_vector& vector);
		vertice_vector& operator=(vertice_vector&& vector) noexcept;

		void release_queue(bool position, bool color, bool indices);

		virtual void push_back(const fan::vec2& position, const fan::color& color, bool queue = false);

		void draw(uint32_t mode);

		void erase(uint_t i, bool queue = false);
		void erase(uint_t begin, uint_t end, bool queue = false);

	protected:

		void write_data();

		uint32_t m_ebo;

		uint_t m_index_restart;

		std::vector<uint32_t> m_indices;

		uint_t m_offset;

	};

	class line_vector : 
		public fan::basic_shape_vector<fan::vec2>, 
		public fan::basic_shape_color_vector<> {
	public:
		
		line_vector(fan::camera& camera);
		line_vector(fan::camera& camera, const fan::mat2& begin_end, const fan::color& color);

		line_vector(const line_vector& vector);
		line_vector(line_vector&& vector) noexcept;

		line_vector& operator=(const line_vector& vector);
		line_vector& operator=(line_vector&& vector) noexcept;

		void resize(uint_t size, const fan::color& color);

		void push_back(const fan::mat2& begin_end, const fan::color& color, bool queue = false);

		void draw();

		void set_position(uint_t i, const fan::mat2& begin_end, bool queue = false);

		void release_queue(bool position, bool color);

		void initialize_buffers();

	private:
		using fan::basic_shape_vector<fan::vec2>::set_position;
		using fan::basic_shape_vector<fan::vec2>::set_size;
	};

	struct triangle_vector : public fan::basic_shape_vector<fan::vec2>, public fan::basic_shape_color_vector<> {

		triangle_vector();
		triangle_vector(const fan::mat3x2& corners, const fan::color& color);
		
		void set_position(uint_t i, const fan::mat3x2& corners);
		void push_back(const fan::mat3x2& corners, const fan::color& color);

		void draw();

	private:

	};

	class rectangle_vector : 
		public fan::basic_shape_vector<fan::vec2>, 
		public fan::basic_shape_color_vector<>, 
		public fan::basic_shape_velocity_vector<fan::vec2> {
	public:

		rectangle_vector(fan::camera& camera);
		rectangle_vector(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color);

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

		void draw(uint_t i = fan::uninitialized);

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

	// fix size
	class sprite_vector : 
		public fan::basic_shape_vector<fan::vec2>,
		public fan::basic_shape_velocity_vector<fan::vec2>,
		public fan::texture_handler {
	public:

		sprite_vector(fan::camera& camera, const std::string& path);
		sprite_vector(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0);
		sprite_vector(const sprite_vector& vector);
		sprite_vector(sprite_vector&& vector) noexcept;
		~sprite_vector();

		sprite_vector& operator=(const sprite_vector& vector);
		sprite_vector& operator=(sprite_vector&& vector) noexcept;

		void initialize_buffers();

		void push_back(const fan::vec2& position, const fan::vec2& size = 0, bool queue = false);

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

	struct particle {
		fan::vec2 m_velocity;
		fan::timer<> m_timer; // milli
	};

	class particles : public fan_2d::rectangle_vector {
	public:

		particles(fan::camera& camera);

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

	fan_2d::line_vector create_grid(fan::camera& camera, const fan::vec2i& block_size, const fan::vec2i& grid_size, const fan::color& color);

	namespace collision {
		constexpr auto sign_dr(f32_t _m) {
			return (si_t)(-(_m < 0) | (_m > 0));
		}

		constexpr fan::da_t<f32_t, 2> LineInterLine_fr(fan::da_t<f32_t, 2, 2> src, fan::da_t<f32_t, 2, 2> dst, const fan::da_t<f32_t, 2>& normal) {
			f32_t s1_x = 0, s1_y = 0, s2_x = 0, s2_y = 0;
			s1_x = src[1][0] - src[0][0]; s1_y = src[1][1] - src[0][1];
			s2_x = dst[1][0] - dst[0][0]; s2_y = dst[1][1] - dst[0][1];

			const f32_t s = (-s1_y * (src[0][0] - dst[0][0]) + s1_x * (src[0][1] - dst[0][1])) / (-s2_x * s1_y + s1_x * s2_y);
			const f32_t t = (s2_x * (src[0][1] - dst[0][1]) - s2_y * (src[0][0] - dst[0][0])) / (-s2_x * s1_y + s1_x * s2_y);

			if (s < 0 || s > 1 || t < 0 || t > 1)
				return FLT_MAX;

			si_t signy = sign_dr(normal.gfne());
			if (fan::dcom_fr(signy > 0, src[1][!!normal[1]], dst[0][!!normal[1]]))
				return FLT_MAX;

			fan::da_t<f32_t, 2> min = dst.min();
			fan::da_t<f32_t, 2> max = dst.max();
			for (uint_t i = 0; i < 2; i++) {
				if (!normal[i])
					continue;
				if (src[0][i ^ 1] == min[i ^ 1])
					return FLT_MAX;
				if (src[0][i ^ 1] == max[i ^ 1])
					return FLT_MAX;
			}

			return { src[0][0] + (t * s1_x), src[0][1] + (t * s1_y) };
		}

		constexpr fan::da_t<uint_t, 2> GetPointsTowardsVelocity2(const fan::da_t<f_t, 2>& vel) {
			if (vel[0] >= 0)
				if (vel[1] >= 0)
					return { 2, 1 };
				else
					return { 0, 3 };
			else
				if (vel[1] >= 0)
					return { 0, 3 };
				else
					return { 2, 1 };
		}

		constexpr fan::da_t<uint_t, 3> GetPointsTowardsVelocity3(const fan::da_t<f32_t, 2>& vel) {
			if (vel[0] >= 0)
				if (vel[1] >= 0)
					return { 2, 1, 3 };
				else
					return { 0, 3, 1 };
			else
				if (vel[1] >= 0)
					return { 0, 3, 2 };
				else
					return { 2, 1, 0 };
		}

		constexpr void calculate_velocity(const fan::da_t<f32_t, 2>& spos, const fan::da_t<f32_t, 2>& svel, const fan::da_t<f32_t, 2>& dpos, const fan::da_t<f32_t, 2>& dvel, const fan::da_t<f32_t, 2>& normal, f32_t sign, fan::da_t<f32_t, 2>& lvel, fan::da_t<f32_t, 2>& nvel) {
			fan::da_t<f32_t, 2, 2> sline = { spos, spos + svel };
			fan::da_t<f32_t, 2, 2> dline = { dpos, dpos + dvel };
			fan::da_t<f32_t, 2> inter = LineInterLine_fr(sline, dline, normal);
			if (inter == FLT_MAX)
				return;
			fan::da_t<f32_t, 2> tvel = (inter - spos) * sign;
			if (tvel.abs() >= lvel.abs())
				return;
			nvel = svel * sign - tvel;
			lvel = tvel;
			nvel[0] = normal[1] ? nvel[0] : 0;
			nvel[1] = normal[0] ? nvel[1] : 0;
		}

		constexpr fan::da_t<f32_t, 4, 2> Math_SquToQuad_fr(const fan::da_t<f32_t, 2, 2>& squ) {
			return fan::da_t<f32_t, 4, 2>{
				fan::da_t<f32_t, 2>(squ[0]),
					fan::da_t<f32_t, 2>(squ[1][0], squ[0][1]),
					fan::da_t<f32_t, 2>(squ[0][0], squ[1][1]),
					fan::da_t<f32_t, 2>(squ[1])
			};
		}

		constexpr auto get_cross(const fan::da_t<f32_t, 2>& a, const fan::da_t<f32_t, 3>& b) {
			//return fan::da_t<f32_t, 2>{a[1] * b[2], b[2] * a[0]}; // could be wrong
			return cross(fan::da_t<f32_t, 3>{ a[0], a[1], 0 }, b);
		}

		template <
			template <typename, std::size_t, std::size_t> typename inner_da_t,
			template <typename, std::size_t> typename outer_da_t, std::size_t n
		>
		constexpr fan::da_t<fan::da_t<f32_t, 2>, n> get_normals(const outer_da_t<inner_da_t<f32_t, 2, 2>, n>& lines) {
			fan::da_t<fan::da_t<f32_t, 2>, n> normals = { 0 };
			for (uint_t i = 0; i < n; i++) {
				normals[i] = get_cross(lines[i][1] - lines[i][0], fan::da_t<f32_t, 3>(0, 0, 1));
			}
			return normals;
		}

		static uint8_t ProcessCollision_fl(fan::da_t<f32_t, 2, 2>& pos, fan::da_t<f32_t, 2>& vel, const std::vector<fan::vec2>& wall_positions, const std::vector<fan::vec2>& wall_sizes) {
			fan::da_t<f32_t, 2> pvel = vel;

			if (!pvel[0] && !pvel[1])
				return 0;

			fan::da_t<f32_t, 4, 2> ocorn = Math_SquToQuad_fr(pos);
			fan::da_t<f32_t, 4, 2> ncorn = ocorn + pvel;

			fan::da_t<uint_t, 3> ptv3 = GetPointsTowardsVelocity3(pvel);
			fan::da_t<uint_t, 3> ntv3 = GetPointsTowardsVelocity3(-pvel);

			fan::da_t<uint_t, 4, 2> li = { fan::da_t<uint_t, 2>{0, 1}, fan::da_t<uint_t, 2>{1, 3}, fan::da_t<uint_t, 2>{3, 2}, fan::da_t<uint_t, 2>{2, 0} };

			const static auto normals = get_normals(
				fan::da_t<fan::da_t<f32_t, 2, 2>, 4>{
					fan::da_t<f32_t, 2, 2>{fan::da_t<f32_t, 2>{ 0, 0 }, fan::da_t<f32_t, 2>{ 1, 0 }},
					fan::da_t<f32_t, 2, 2>{fan::da_t<f32_t, 2>{ 1, 0 }, fan::da_t<f32_t, 2>{ 1, 1 }},
					fan::da_t<f32_t, 2, 2>{fan::da_t<f32_t, 2>{ 1, 1 }, fan::da_t<f32_t, 2>{ 0, 1 }},
					fan::da_t<f32_t, 2, 2>{fan::da_t<f32_t, 2>{ 0, 1 }, fan::da_t<f32_t, 2>{ 0, 0 }},
			});

			fan::da_t<f32_t, 2> lvel = pvel;
			fan::da_t<f32_t, 2> nvel = 0;
			for (uint_t iwall = 0; iwall < wall_positions.size(); iwall++) {
				fan::da_t<f32_t, 4, 2> bcorn = Math_SquToQuad_fr({ wall_positions[iwall], wall_positions[iwall] + wall_sizes[iwall] });

				/* step -1 */
				for (uint_t i = 0; i < 4; i++) {
					for (uint_t iline = 0; iline < 4; iline++) {
						calculate_velocity(fan::da_t<f32_t, 2, 2>(ocorn[li[i][0]], ocorn[li[i][1]]).avg(), pvel, bcorn[li[iline][0]], bcorn[li[iline][1]] - bcorn[li[iline][0]], normals[iline], 1, lvel, nvel);
					}
				}

				/* step 0 and step 1*/
				for (uint_t i = 0; i < 3; i++) {
					for (uint_t iline = 0; iline < 4; iline++) {
						calculate_velocity(ocorn[ptv3[i]], ncorn[ptv3[i]] - ocorn[ptv3[i]], bcorn[li[iline][0]], bcorn[li[iline][1]] - bcorn[li[iline][0]], normals[iline], 1, lvel, nvel);
						calculate_velocity(bcorn[ntv3[i]], -pvel, ocorn[li[iline][0]], ocorn[li[iline][1]] - ocorn[li[iline][0]], normals[iline], -1, lvel, nvel);
					}
				}
			}

			pos += lvel;
			vel = nvel;

			return 1;
		}

		#define ProcessCollision_dl(pos_m, vel_m, wall_positions_m, wall_sizes_m) \
			while(ProcessCollision_fl(pos_m, vel_m, wall_positions_m, wall_sizes_m))

		inline void rectangle_collision(fan::window& window, fan_2d::rectangle& player, const fan_2d::rectangle_vector& walls) {
			const fan::da_t<f32_t, 2> size = player.get_size();
			const fan::da_t<f32_t, 2> base = player.get_velocity();
			fan::da_t<f32_t, 2> velocity = base * window.get_delta_time();
			const fan::da_t<f32_t, 2> old_position = player.get_position() - velocity;
			fan::da_t<f32_t, 2, 2> my_corners(old_position, old_position + size);
			ProcessCollision_dl(my_corners, velocity, walls.get_positions(), walls.get_sizes());
			player.set_position(my_corners[0]);
		}

	}

	namespace gui {
		
		fan::vec2 get_resize_movement_offset(fan::window& window);

		void add_resize_callback(fan::window& window, fan::vec2& position);

		struct rectangle : public fan_2d::rectangle {

			rectangle(fan::camera& camera);
			rectangle(fan::camera& camera,const fan::vec2& position, const fan::vec2& size, const fan::color& color);

		};

		struct rectangle_vector : public fan_2d::rectangle_vector {

			rectangle_vector(fan::camera& camera);
			rectangle_vector(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color);

		};

		struct sprite : public fan_2d::sprite {

			sprite(fan::camera& camera);
			// scale with default is sprite size
			sprite(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0, f_t transparency = 1);
			sprite(fan::camera& camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size = 0, f_t transparency = 1);

		};

		struct sprite_vector : public fan_2d::sprite_vector {

			sprite_vector(fan::camera& camera, const std::string& path);
			sprite_vector(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0);

		};

		class rounded_rectangle : public fan_2d::vertice_vector {
		public:

		    static constexpr f_t segments = 4 * 20; // corners * random

			rounded_rectangle(fan::camera& camera);
			rounded_rectangle(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color);

			void push_back(const fan::vec2& position, const fan::vec2& size, const fan::color& color, bool queue = false);

			fan::vec2 get_position(uint_t i) const;
			void set_position(uint_t i, const fan::vec2& position);

			fan::vec2 get_size(uint_t i) const;
			void set_size(uint_t i); // ?

			void draw();

		private:

			using fan_2d::vertice_vector::push_back;
			std::vector<fan::vec2> m_position;
			std::vector<fan::vec2> m_size;
			std::vector<uint_t> data_offset;

		};

		namespace font_properties {
			constexpr uint16_t max_ascii(256);
			constexpr uint_t max_font_size(1024);
			constexpr uint_t new_line(64);
			constexpr f_t gap_multiplier(1.25);

			constexpr fan::color default_text_color(1);
			constexpr f_t edge(0.1);

			constexpr f_t get_gap_size(unsigned int advance) {
				return advance * (1.0 / gap_multiplier);
			}

		}

		class text_renderer : protected fan::basic_shape_color_vector_vector<0, GL_SHADER_STORAGE_BUFFER>, protected fan::basic_shape_color_vector_vector<4, GL_SHADER_STORAGE_BUFFER> {
		public:

			using text_color_t = fan::basic_shape_color_vector_vector<0, GL_SHADER_STORAGE_BUFFER>;
			using outline_color_t = fan::basic_shape_color_vector_vector<4, GL_SHADER_STORAGE_BUFFER>;

			text_renderer(fan::camera& camera);
			text_renderer(fan::camera& camera, const std::string& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color = fan::color(-1, -1, -1, 0), bool queue = false);
			~text_renderer();

			fan::vec2 get_position(uint_t i) const;
			
			void set_position(uint_t i, const fan::vec2& position, bool queue = false);

			f_t get_font_size(uint_t i) const;
			void set_font_size(uint_t i, f_t font_size, bool queue = false);
			void set_text(uint_t i, const std::string& text, bool queue = false);
			void set_text_color(uint_t i, const fan::color& color, bool queue = false);
			void set_outline_color(uint_t i, const fan::color& color, bool queue = false);

			fan::vec2 get_text_size(const std::string& text, f_t font_size) const;

			f_t get_font_height_max(uint32_t font_size);

			std::string get_text(uint_t i) const;

			f_t convert_font_size(f_t font_size) const;

			void free_queue();
			void insert(uint_t i, const std::string& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color = fan::uninitialized, bool queue = false);
			void push_back(const std::string& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color = fan::uninitialized, bool queue = false);

			void draw() const;

			void erase(uint_t i, bool queue = false);

			uint_t size() const;

		private:

			uint_t get_character_offset(uint_t i, bool special);
			
			std::vector<fan::vec2> get_vertices(uint_t i);
			std::vector<fan::vec2> get_texture_coordinates(uint_t i);

			void load_characters(uint_t i, fan::vec2 position, const std::string& text, bool edit, bool insert);

			void edit_letter_data(uint_t i, uint_t j, const char letter, const fan::vec2& position, int& advance, f_t converted_font_size);
			void insert_letter_data(uint_t i, const char letter, const fan::vec2& position, int& advance, f_t converted_font_size);
			void write_letter_data(uint_t i, const char letter, const fan::vec2& position, int& advance, f_t converted_font_size);

			void write_vertices();
			void write_texture_coordinates();
			void write_font_sizes();

			void write_data();

			fan::shader m_shader;

			uint32_t m_texture;

			uint32_t m_vao;

			uint32_t m_vertex_vbo;
			uint32_t m_texture_vbo;

			uint32_t m_letter_ssbo;
			
			f_t m_original_font_size;
			fan::vec2ui m_original_image_size;

			std::unordered_map<uint16_t, fan::io::file::font_t> m_font;

			std::vector<std::string> m_text;

			std::vector<fan::vec2> m_position;

			std::vector<std::vector<f32_t>> m_font_size;
			std::vector<std::vector<fan::vec2>> m_vertices;
			std::vector<std::vector<fan::vec2>> m_texture_coordinates;

			fan::window& m_window;
			fan::camera& m_camera;

		};

		//inline fan_2d::gui::text_renderer global_text_renderer("", 0, 1, 32);

		//struct text_draw_info {
		//	uint_t id;
		//	fan::vec2 position;
		//	fan::color color;
		//	f_t font_size;
		//};

		//namespace static_function_variables {
		//	inline std::map<std::string, text_draw_info> text_draw_strings;
		//}

		//static void draw_text_basic(const std::string& map_string, const std::string& text, const fan::vec2& position, const fan::color& color, f_t font_size) {
		//	auto found = fan_2d::gui::static_function_variables::text_draw_strings.find(map_string);

		//	if (found == fan_2d::gui::static_function_variables::text_draw_strings.end()) {
		//		fan_2d::gui::static_function_variables::text_draw_strings.insert(std::make_pair(map_string, text_draw_info{ fan_2d::gui::global_text_renderer.size(), position, color, font_size }));
		//		fan_2d::gui::global_text_renderer.push_back(text, position, color, font_size);
		//	}
		//	else {
		//		const auto& object = fan_2d::gui::static_function_variables::text_draw_strings[map_string];

		//		fan_2d::gui::global_text_renderer.set_position(object.id, position, true);
		//		fan_2d::gui::global_text_renderer.set_text_color(object.id, color, true);
		//		fan_2d::gui::global_text_renderer.set_font_size(object.id, font_size, true);
		//		fan_2d::gui::global_text_renderer.set_text(object.id, text, true);

		//		fan_2d::gui::global_text_renderer.free_queue();
		//	}
		//}

		//// will update when std::source_location is implemented
		//#define draw_text(text, position, color, font_size) draw_text_basic(std::string(__FILE__) + ':' + std::to_string(__LINE__) + ':' + text, text, position, color, font_size);

	}
}

namespace fan_3d {

	namespace shader_paths {
		constexpr auto triangle_vector_vs("include/fan/glsl/3D/terrain_generator.vs");
		constexpr auto triangle_vector_fs("include/fan/glsl/3D/terrain_generator.fs");

		constexpr auto shape_vector_vs("include/fan/glsl/3D/shape_vector.vs");
		constexpr auto shape_vector_fs("include/fan/glsl/3D/shape_vector.fs");

		constexpr auto model_vs("include/fan/glsl/3D/models.vs");
		constexpr auto model_fs("include/fan/glsl/3D/models.fs");

		constexpr auto animation_vs("include/fan/glsl/3D/animation.vs");
		constexpr auto animation_fs("include/fan/glsl/3D/animation.fs");

		constexpr auto skybox_vs("include/fan/glsl/3D/skybox.vs");
		constexpr auto skybox_fs("include/fan/glsl/3D/skybox.fs");
		constexpr auto skybox_model_vs("include/fan/glsl/3D/skybox_model.vs");
		constexpr auto skybox_model_fs("include/fan/glsl/3D/skybox_model.fs");
	}

	void add_camera_rotation_callback(fan::camera& camera);

	class line_vector : public fan::basic_shape_vector<fan::vec3>, public fan::basic_shape_color_vector<> {
	public:

		line_vector(fan::camera& camera);
		line_vector(fan::camera& camera, const fan::mat2x3& begin_end, const fan::color& color);

		void push_back(const fan::mat2x3& begin_end, const fan::color& color, bool queue = false);

		void draw();

		void set_position(uint_t i, const fan::mat2x3 begin_end, bool queue = false);
		
		void release_queue(bool position, bool color);

	private:

		using fan::basic_shape_vector<fan::vec3>::set_position;
		using fan::basic_shape_vector<fan::vec3>::set_size;

	};

	using triangle_vertices_t = fan::vec3;

	class terrain_generator : public fan::basic_shape_color_vector<> {
	public:

		terrain_generator(fan::camera& camera, const std::string& path, const f32_t texture_scale, const fan::vec3& position, const fan::vec2ui& map_size, f_t triangle_size, const fan::vec2& mesh_size);
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

		fan::window& m_window;
		fan::camera& m_camera;

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

	class square_vector : protected fan::basic_shape_vector<fan::vec3>, public fan::basic_shape_color_vector<> {
	public:

		square_vector(fan::camera& camera, const std::string& path, uint_t block_size);
		square_vector(fan::camera& camera, const fan::color& color, uint_t block_size);
		~square_vector();

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

		unsigned int m_texture;
		unsigned int m_texture_ssbo;
		unsigned int m_texture_id_ssbo;

		fan::vec2i block_size;
		fan::vec2i m_amount_of_textures;

		std::vector<int> m_textures;

	};

	class skybox {
	public:
		skybox(
			fan::camera& camera,
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
		fan::camera& m_camera;
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
		model(fan::camera& camera);
		model(fan::camera& camera, const std::string& path, const fan::vec3& position, const fan::vec3& size);

		void draw();

		fan::vec3 get_position();
		void set_position(const fan::vec3& position);

		fan::vec3 get_size();
		void set_size(const fan::vec3& size);

	private:
		fan::shader m_shader;

		fan::vec3 m_position;
		fan::vec3 m_size;

		fan::window& m_window;
		fan::camera m_camera;

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

	/*
	f_t travel_distance = fan_2d::manhattan_distance(start, end); \
		if (!(start == end)) \                                                     
			while(grid_raycast_single(raycast, block_size) && travel_distance < fan_2d::manhattan_distance(src, raycast.grid * block_size))
	*/

	#define d_grid_raycast_2d(start, end, raycast, block_size) \
		fan::grid_raycast_s<fan::vec2> raycast = { grid_direction(end, start), start, fan::vec2() }; \
		f_t _private_travel_distance = fan_2d::distance((start / block_size).floored(), (end / block_size).floored()); \
		if (!(start == end)) \
			while(grid_raycast_single(raycast, block_size) && _private_travel_distance >= fan_2d::distance((start / block_size).floored(), raycast.grid))

	#define d_grid_raycast_3d(start, end, raycast, block_size) \
		fan::grid_raycast_s<fan::vec3> raycast = { grid_direction(end, start), start, fan::vec3() }; \
		if (!(start == end)) \
			while(grid_raycast_single(raycast, block_size))

	static void draw_2d(const std::function<void()>& function_) {
		glDisable(GL_DEPTH_TEST);
		function_();
		glEnable(GL_DEPTH_TEST);
	}
}