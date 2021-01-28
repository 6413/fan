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

		static constexpr f_t sensitivity = 0.05f;

		static constexpr f_t max_yaw = 180;
		static constexpr f_t max_pitch = 89;

		static constexpr f_t gravity = 500;
		static constexpr f_t jump_force = 100;

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

		basic_shape_position();

		basic_shape_position(const _Vector& position);

		basic_shape_position(const basic_shape_position& vector);
		basic_shape_position(basic_shape_position&& vector) noexcept;

		basic_shape_position& operator=(const basic_shape_position& vector);
		basic_shape_position& operator=(basic_shape_position&& vector);

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void reserve(uint_t new_size);
		enable_function_for_vector void resize(uint_t new_size);

		enable_function_for_vector std::vector<_Vector> get_positions() const;

		enable_function_for_vector void set_positions(const std::vector<_Vector>& positions);

		enable_function_for_vector _Vector get_position(uint_t i) const;
		enable_function_for_vector void set_position(uint_t i, const _Vector& position, bool queue = false);

		enable_function_for_vector void erase(uint_t i, bool queue = false);
		enable_function_for_vector void erase(uint_t begin, uint_t end, bool queue = false);

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector _Vector get_position() const;
		enable_function_for_non_vector void set_position(const _Vector& position);

		// -----------------------------------------------------

	protected:

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void initialize_buffers(bool divisor);

		enable_function_for_vector void basic_push_back(const _Vector& position, bool queue = false);

		enable_function_for_vector void edit_data(uint_t i);
		enable_function_for_vector void write_data();

		// -----------------------------------------------------

		std::conditional_t<enable_vector, std::vector<_Vector>, _Vector> m_position;

	};
 
	template <bool enable_vector, typename _Vector, uint_t layout_location = 2, opengl_buffer_type buffer_type = opengl_buffer_type::buffer_object>
	class basic_shape_size : public glsl_location_handler<layout_location, buffer_type> {
	public:

		basic_shape_size();
		basic_shape_size(const _Vector& size);

		basic_shape_size(const basic_shape_size& vector);
		basic_shape_size(basic_shape_size&& vector) noexcept;

		basic_shape_size& operator=(const basic_shape_size& vector);
		basic_shape_size& operator=(basic_shape_size&& vector) noexcept;

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void reserve(uint_t new_size);
		enable_function_for_vector void resize(uint_t new_size);

		enable_function_for_vector std::vector<_Vector> get_sizes() const;

		enable_function_for_vector _Vector get_size(uint_t i) const;
		enable_function_for_vector void set_size(uint_t i, const _Vector& size, bool queue = false);

		enable_function_for_vector void erase(uint_t i, bool queue = false);
		enable_function_for_vector void erase(uint_t begin, uint_t end, bool queue = false);

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector _Vector get_size() const;
		enable_function_for_non_vector void set_size(const _Vector& size);

		// -----------------------------------------------------

	protected:

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void basic_push_back(const _Vector& size, bool queue = false);

		enable_function_for_vector void edit_data(uint_t i);
		enable_function_for_vector void write_data();

		enable_function_for_vector void initialize_buffers(bool divisor);

		// -----------------------------------------------------

		std::conditional_t<enable_vector, std::vector<_Vector>, _Vector> m_size;

	};

	template <bool enable_vector, uint_t layout_location = 0, opengl_buffer_type buffer_type = opengl_buffer_type::buffer_object>
	class basic_shape_color_vector : public glsl_location_handler<layout_location, buffer_type> {
	public:

		basic_shape_color_vector();
		basic_shape_color_vector(const basic_shape_color_vector& vector);
		basic_shape_color_vector(basic_shape_color_vector&& vector) noexcept;

		basic_shape_color_vector& operator=(const basic_shape_color_vector& vector);
		basic_shape_color_vector& operator=(basic_shape_color_vector&& vector) noexcept;

		basic_shape_color_vector(const fan::color& color);

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void reserve(uint_t new_size);
		enable_function_for_vector void resize(uint_t new_size, const fan::color& color);
		enable_function_for_vector void resize(uint_t new_size);

		enable_function_for_vector fan::color get_color(uint_t i);

		enable_function_for_vector void set_color(uint_t i, const fan::color& color, bool queue = false) {
			this->m_color[i] = color;
			if (!queue) {
				this->edit_data(i);
			}
		}

		enable_function_for_vector void erase(uint_t i, bool queue = false);
		enable_function_for_vector void erase(uint_t begin, uint_t end, bool queue = false);

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector fan::color get_color() const;
		enable_function_for_non_vector void set_color(const fan::color& color);

		// -----------------------------------------------------

	protected:

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void basic_push_back(const fan::color& color, bool queue = false);
	
		enable_function_for_vector void edit_data(uint_t i);
		enable_function_for_vector void edit_data(void* data, uint_t offset, uint_t byte_size);

		enable_function_for_vector void write_data();

		enable_function_for_vector void initialize_buffers(bool divisor);

		// -----------------------------------------------------

		std::conditional_t<enable_vector, std::vector<fan::color>, fan::color> m_color;

	};

	template <bool enable_vector, typename _Vector>
	class basic_shape_velocity {
	public:

		basic_shape_velocity();
		basic_shape_velocity(const _Vector& velocity);

		basic_shape_velocity(const basic_shape_velocity& vector);
		basic_shape_velocity(basic_shape_velocity&& vector) noexcept;

		basic_shape_velocity& operator=(const basic_shape_velocity& vector);
		basic_shape_velocity& operator=(basic_shape_velocity&& vector) noexcept;

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector _Vector get_velocity(uint_t i) const;
		enable_function_for_vector void set_velocity(uint_t i, const _Vector& velocity);

		enable_function_for_vector void reserve(uint_t new_size);
		enable_function_for_vector void resize(uint_t new_size);

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector _Vector get_velocity() const;
		enable_function_for_non_vector void set_velocity(const _Vector& velocity);

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
		
		basic_shape(fan::camera& camera);
		basic_shape(fan::camera& camera, const fan::shader& shader);
		basic_shape(fan::camera& camera, const fan::shader& shader, const _Vector& position, const _Vector& size);

		basic_shape(const basic_shape& vector);
		basic_shape(basic_shape&& vector) noexcept;

		basic_shape& operator=(const basic_shape& vector);
		basic_shape& operator=(basic_shape&& vector) noexcept;

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void reserve(uint_t new_size);
		enable_function_for_vector void resize(uint_t new_size);

		enable_function_for_vector uint_t size() const;

		// -----------------------------------------------------

		f_t get_delta_time() const;

		fan::camera& m_camera;

	protected:

		// ----------------------------------------------------- vector enabled functions

		enable_function_for_vector void basic_push_back(const _Vector& position, const _Vector& size, bool queue = false);

		enable_function_for_vector void erase(uint_t i, bool queue = false);
		enable_function_for_vector void erase(uint_t begin, uint_t end, bool queue = false);

		enable_function_for_vector void edit_data(uint_t i, bool position, bool size);

		enable_function_for_vector void write_data(bool position, bool size);

		enable_function_for_vector void basic_draw(unsigned int mode, uint_t count, uint_t primcount, uint_t i = fan::uninitialized) const;

		// -----------------------------------------------------

		// ----------------------------------------------------- non vector enabled functions

		enable_function_for_non_vector void basic_draw(unsigned int mode, uint_t count) const;

		// -----------------------------------------------------

		fan::shader m_shader;

		fan::window& m_window;

	};

	template <typename _Vector>
	class basic_vertice_vector : 
		public basic_shape_position<true, _Vector>, 
		public basic_shape_color_vector<true>, 
		public basic_shape_velocity<true, _Vector>,
	    public vao_handler {
	public:

		basic_vertice_vector(fan::camera& camera, const fan::shader& shader);
		basic_vertice_vector(fan::camera& camera, const fan::shader& shader, const fan::vec2& position, const fan::color& color);

		basic_vertice_vector(const basic_vertice_vector& vector) ;
		basic_vertice_vector(basic_vertice_vector&& vector) noexcept;

		basic_vertice_vector& operator=(const basic_vertice_vector& vector);
		basic_vertice_vector& operator=(basic_vertice_vector&& vector);

		uint_t size() const;

		void erase(uint_t i, bool queue = false);
		void erase(uint_t begin, uint_t end, bool queue = false);

		fan::window& m_window;

	protected:

		void basic_push_back(const _Vector& position, const fan::color& color, bool queue = false);

		void edit_data(uint_t i, bool position, bool color);

		void write_data(bool position, bool color);

		void basic_draw(unsigned int mode, uint_t count) const;

		fan::shader m_shader;

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

	struct line : 
		protected fan::basic_shape<0, fan::vec2>, 
		public fan::basic_shape_color_vector<0> {

		line(fan::camera& camera);
		line(fan::camera& camera, const fan::mat2& begin_end, const fan::color& color);

		void draw();

		fan::mat2 get_position() const;
		void set_position(const fan::mat2& begin_end);

	};

	struct rectangle : 
		public fan::basic_shape<0, fan::vec2>, 
		public fan::basic_shape_color_vector<0>, 
		public fan::basic_shape_velocity<0, fan::vec2> {

		rectangle(fan::camera& camera);
		rectangle(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, const fan::color& color);

		rectangle(const rectangle& single);
		rectangle(rectangle&& single) noexcept;

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
		sprite(fan::camera& camera);

		// size with default is the size of the image
		sprite(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0, f_t transparency = 1);
		sprite(fan::camera& camera, unsigned char* pixels, const fan::vec2& position, const fan::vec2i& size = 0, f_t transparency = 1);

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

		vertice_vector(fan::camera& camera, uint_t index_restart = UINT32_MAX);
		vertice_vector(fan::camera& camera, const fan::vec2& position, const fan::color& color, uint_t index_restart);
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
		
		line_vector(fan::camera& camera);
		line_vector(fan::camera& camera, const fan::mat2& begin_end, const fan::color& color);

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

		sprite_vector(fan::camera& camera, const std::string& path);
		sprite_vector(fan::camera& camera, const std::string& path, const fan::vec2& position, const fan::vec2& size = 0);
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

		rounded_rectangle(fan::camera& camera);
		rounded_rectangle(fan::camera& camera, const fan::vec2& position, const fan::vec2& size, f_t radius, const fan::color& color);

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

		namespace font_properties {
			constexpr f_t new_line(70);
			constexpr f_t gap_multiplier(1);

			constexpr fan::color default_text_color(1);

			constexpr f_t space_width(15);

			constexpr f_t get_new_line(f_t font_size) {
				return new_line * font_size;
			}

		}

		class text_renderer : protected fan::basic_shape_color_vector_vector<0, GL_SHADER_STORAGE_BUFFER>, protected fan::basic_shape_color_vector_vector<4, GL_SHADER_STORAGE_BUFFER> {
		public:

			using text_color_t = fan::basic_shape_color_vector_vector<0, GL_SHADER_STORAGE_BUFFER>;
			using outline_color_t = fan::basic_shape_color_vector_vector<4, GL_SHADER_STORAGE_BUFFER>;

			text_renderer(fan::camera& camera);
			text_renderer(fan::camera& camera, const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f_t font_size, const fan::color& outline_color = fan::color(-1, -1, -1, 0), bool queue = false);
			~text_renderer();

			fan::vec2 get_position(uint_t i) const;
			
			void set_position(uint_t i, const fan::vec2& position, bool queue = false);

			f_t get_font_size(uint_t i) const;
			void set_font_size(uint_t i, f_t font_size, bool queue = false);
			void set_text(uint_t i, const fan::fstring& text, bool queue = false);
			void set_text_color(uint_t i, const fan::color& color, bool queue = false);
			void set_outline_color(uint_t i, const fan::color& color, bool queue = false);

			fan::io::file::font_t get_letter_info(fan::fstring::value_type c, f_t font_size) const;
			fan::vec2 get_text_size(const fan::fstring& text, f_t font_size) const;
			fan::vec2 get_text_size_original(const fan::fstring& text, f_t font_size) const;

			f_t get_lowest(f_t font_size) const;
			f_t get_highest(f_t font_size) const;

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

			std::unordered_map<uint16_t, fan::io::file::font_t> m_font;

			fan::camera& m_camera;

		private:

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

			uint32_t m_texture;

			uint32_t m_vao;

			uint32_t m_vertex_vbo;
			uint32_t m_texture_vbo;

			uint32_t m_letter_ssbo;
			
			f_t m_original_font_size;
			fan::vec2ui m_original_image_size;

			std::vector<fan::fstring> m_text;

			std::vector<fan::vec2> m_position;

			std::vector<std::vector<f32_t>> m_font_size;
			std::vector<std::vector<fan::vec2>> m_vertices;
			std::vector<std::vector<fan::vec2>> m_texture_coordinates;

			fan::window& m_window;

		};

		namespace text_box_properties {

			static inline int blink_speed(500); // ms
			static inline fan::color cursor_color(fan::colors::white);

		}

		template <typename T>
		class basic_text_box {
		public:

			basic_text_box(fan::camera& camera, const fan::fstring& text, const fan::vec2& position, const fan::color& text_color, f_t font_size);

			void set_input_callback(uint_t i) {

				m_text_input.m_offset.emplace_back(0);
				m_text_input.m_str.emplace_back(this->get_text(i));

				const auto& text = this->get_text(i);

				m_new_lines.emplace_back(std::count_if(text.begin(), text.end(), [](const fan::fstring::value_type c) { return c == '\n'; }));

				m_current_cursor_line.emplace_back(m_new_lines[m_new_lines.size() - 1]);

				m_callable.emplace_back(i);

				m_text_visual_input.m_cursor.push_back(fan::mat2(), text_box_properties::cursor_color);
				m_text_visual_input.m_timer.emplace_back(fan::timer<>(fan::timer<>::start(), text_box_properties::blink_speed));
				m_text_visual_input.m_visible.emplace_back(true);

				update_cursor_position(i);

				m_tr.m_camera.m_window.set_keys_callback([&](fan::fstring::value_type key) {
					for (auto j : m_callable) {
						handle_input(j, key);
					}
				});
			}

			fan::vec2 get_border_size(uint_t i) const {
				return m_border_size[i];
			}

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

			// returns begin and end of cursor points
			fan::mat2 get_cursor_position(uint_t i, uint_t beg = fan::uninitialized, uint_t n = fan::uninitialized) const {

				const auto& str = this->get_text(i);

				const auto font_size = this->get_font_size(i);

				auto converted = this->m_tr.convert_font_size(font_size);

				f_t x = 0;
				f_t y = 0;

				const std::size_t n_ = n == (std::size_t)-1 ? str.size() : n;

				for (std::size_t j = (beg == (std::size_t)-1 ? 0 : beg); j < n_; j++) {
					if (str[j] == '\n') {
						x = 0;
						y += fan_2d::gui::font_properties::get_new_line(m_tr.convert_font_size(m_tr.get_font_size(i)));
					}
					x += m_tr.m_font.find(str[j])->second.m_advance * converted;
				}

				const fan::vec2 position(this->get_position(i));
				const fan::vec2 border_size(this->get_border_size(i));

				x += position.x + border_size.x * 0.5;

				f_t test = (m_new_lines[i] - m_current_cursor_line[i]) * fan_2d::gui::font_properties::get_new_line(m_tr.convert_font_size(m_tr.get_font_size(i)));

				return fan::mat2(
					fan::vec2(x, y + position.y + border_size.y * 0.5), 
					fan::vec2(x, position.y + get_size(i).y - border_size.y * 0.5 - test)
				);
			}

			void update_cursor_position(uint_t i) {
				m_text_visual_input.m_cursor.set_position(i, this->get_cursor_position(i, 0, this->get_text(i).size() + m_text_input.m_offset[i]));
			}

			void set_position(uint_t i, const fan::vec2& position, bool queue = false) {
				m_rv.set_position(i, position, queue);
				m_tr.set_position(i, position + m_border_size[i].y / 2, queue);
			}

			void set_text(uint_t i, const fan::fstring& text, bool queue = false) {
				m_tr.set_text(i, text, queue);

				this->update_box(i);

				update_cursor_position(i);
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

			void set_font_size(uint_t i, f_t font_size, bool queue = false) {
				m_tr.set_font_size(i, font_size);

				update_box(i, queue);

				update_cursor_position(i);
			}

			void draw() {

				for (int i = 0; i < m_text_visual_input.m_cursor.size(); i++) {

					if (m_text_visual_input.m_timer[i].finished()) {
						m_text_visual_input.m_visible[i] = !m_text_visual_input.m_visible[i];
						m_text_visual_input.m_timer[i].restart();
					}
				}

				fan::draw_2d([&] {
					m_rv.draw();
					m_tr.draw();

					if (!m_text_visual_input.m_cursor.size()) {
						return;
					}

					if (m_callable.size() == m_text_visual_input.m_cursor.size()) {
						if (m_text_visual_input.m_visible[0]) {
							m_text_visual_input.m_cursor.draw();
						}	
					}
					else { // in case we dont want to draw input for some window
						for (int i = 0; i < m_callable.size(); i++) {
							if (m_text_visual_input.m_visible[i]) {
								m_text_visual_input.m_cursor.draw(i);
							}
						}
					}
				});
			}

			bool inside(uint_t i) const {
				return m_rv.inside(i);
			}

			void on_touch(std::function<void(uint_t j)> function) {
				m_on_touch = function;

				m_rv.m_camera.m_window.add_mouse_move_callback([&] {
					for (uint_t i = 0; i < m_rv.size(); i++) {
						if (m_rv.inside(i)) {
							m_on_touch(i);
						}
					}
				});
			}

			void on_touch(uint_t i, const std::function<void(uint_t j)>& function) {
				m_on_touch = function;

				m_rv.m_camera.m_window.add_mouse_move_callback([&] {
					if (m_rv.inside(i)) {
						m_on_touch(i);
					}
				});
			}

			void on_exit(std::function<void(uint_t j)> function) {
				m_on_exit = function;

				m_rv.m_camera.m_window.add_mouse_move_callback([&] {
					for (uint_t i = 0; i < m_rv.size(); i++) {
						if (!inside(i)) {
							m_on_exit(i);
						}
					}
				});
			}

			void on_exit(uint_t i, const std::function<void(uint_t j)>& function) {
				m_on_exit = function;

				m_rv.m_camera.m_window.add_mouse_move_callback([&] {
					if (!inside(i)) {
						m_on_exit(i);
					}
				});
			}

			void on_click(std::function<void()> function, uint16_t key = fan::mouse_left) {
				m_on_click = function;

				m_rv.m_camera.m_window.add_key_callback(key, [&] {
					for (uint_t i = 0; i < m_rv.size(); i++) {
						if (m_rv.inside(i)) {
							m_on_click();
						}
					}
				});
			}

			void on_click(uint_t i, const std::function<void()>& function, uint16_t key = fan::mouse_left) {
				m_on_click = function;

				m_rv.m_camera.m_window.add_key_callback(key, [&] {
					if (m_rv.inside(i)) {
						m_on_click();
					}
				});
			}

			void on_release(std::function<void()> function, uint16_t key = fan::mouse_left) {
				m_on_release = function;

				m_rv.m_camera.m_window.add_key_callback(key, [&] {
					for (uint_t i = 0; i < m_rv.size(); i++) {
						if (inside(i)) {
							m_on_release();
						}
					}
				}, true);
			}
			void on_release(uint_t i, const std::function<void()>& function, uint16_t key = fan::mouse_left) {
				m_on_release = function;

				m_rv.m_camera.m_window.add_key_callback(key, [&] {
					if (inside(i)) {
						m_on_release();
					}
				}, true);
			}

			void handle_input(uint_t i, uint_t key) {
				switch (m_tr.m_camera.m_window.get_current_key()) {
					case fan::key_delete: {
						if (m_text_input.m_offset[i] < 0) {
							if (m_text_input.m_str[i][m_text_input.m_str[i].size() + m_text_input.m_offset[i]] == '\n') {
								m_new_lines[i]--;
							}

							m_text_input.m_str[i].erase(m_text_input.m_str[i].end() + m_text_input.m_offset[i]);
							m_text_input.m_offset[i]++;

							this->set_text(i, m_text_input.m_str[i]);
						}
						break;
					}
					case fan::key_backspace:
					{
						if (m_text_input.m_str[i].size()) {
							if (m_text_input.m_str[i][m_text_input.m_str[i].size() + m_text_input.m_offset[i] - 1] == '\n') {
								m_new_lines[i]--;
							}

							m_text_input.m_str[i].erase(m_text_input.m_str[i].end() + m_text_input.m_offset[i] - 1);

							this->set_text(i, m_text_input.m_str[i]);
						}
						break;
					}
					case fan::key_left:
					{

						m_text_input.m_offset[i] = std::clamp(--m_text_input.m_offset[i], -(int64_t)m_text_input.m_str[i].size(), (int64_t)0);

						if (m_text_input.m_str[i][m_text_input.m_str[i].size() + m_text_input.m_offset[i]] == '\n') {
							m_current_cursor_line[i]--;
						}

						update_cursor_position(i);

						break;
					}
					case fan::key_right: {
						m_text_input.m_offset[i] = std::clamp(++m_text_input.m_offset[i], -(int64_t)m_text_input.m_str[i].size(), (int64_t)0);

						if (m_text_input.m_str[i][m_text_input.m_str[i].size() + m_text_input.m_offset[i] - 1] == '\n') {
							m_current_cursor_line[i]++;
						}

						update_cursor_position(i);

						break;
					}
					case fan::key_home:
					{
						m_text_input.m_offset[i] = -(int64_t)m_text_input.m_str[i].size();

						m_current_cursor_line[i] = 0;

						update_cursor_position(i);

						break;
					}
					case fan::key_end:
					{
						m_text_input.m_offset[i] = 0;

						m_current_cursor_line[i] = m_new_lines[i];

						update_cursor_position(i);

						break;
					}
					case fan::key_up:
					case fan::key_down:
					{
						break;
					}
					case fan::key_enter: {

						m_text_input.m_str[i].insert(m_text_input.m_str[i].end() + m_text_input.m_offset[i], '\n');

						m_new_lines[i]++;
						m_current_cursor_line[i]++;

						this->set_text(i, m_text_input.m_str[i]);

						break;
					}
					default:
					{
						m_text_input.m_str[i].insert(m_text_input.m_str[i].end() + m_text_input.m_offset[i], key);
						this->set_text(0, m_text_input.m_str[i]);
					}
				}
			}

		protected:

			fan::vec2 get_updated_size(uint_t i) const {
				f_t h = this->get_lowest(this->get_font_size(i)) + this->get_highest(this->get_font_size(m_border_size.size() - 1));

				if (m_new_lines.size() && m_new_lines[i]) {
					h += fan_2d::gui::font_properties::new_line * m_tr.convert_font_size(m_tr.get_font_size(i)) * m_new_lines[i];
				}

				return fan::vec2(m_tr.get_text(i).empty() ? fan::vec2(0, h) : fan::vec2(m_tr.get_text_size(m_tr.get_text(i), m_tr.get_font_size(i)).x, h)) + m_border_size[i];
			}

			void update_box(uint_t i, bool queue = false) {			
				m_rv.set_size(i, this->get_updated_size(i), queue);
			}

			std::function<void(uint_t i)> m_on_touch;
			std::function<void(uint_t i)> m_on_exit;
			std::function<void()> m_on_click;
			std::function<void()> m_on_release;

			std::vector<fan::vec2> m_border_size;

			T m_rv;
			fan_2d::gui::text_renderer m_tr;

			struct text_input {
				std::vector<int64_t> m_offset;
				std::vector<fan::fstring> m_str;
			};

			struct text_visual_input {

				text_visual_input(fan::camera& camera) : m_cursor(camera) {}
				text_visual_input(fan::camera& camera, const fan::mat2& position, const fan::color& color = text_box_properties::cursor_color) 
						: m_cursor(camera, position, color) {}

				fan_2d::line_vector m_cursor;
				std::vector<fan::timer<>> m_timer;
				std::vector<bool> m_visible;
			};

			text_input m_text_input;
			text_visual_input m_text_visual_input;

			std::vector<uint_t> m_callable;

			std::vector<uint_t> m_new_lines;

			std::vector<uint_t> m_current_cursor_line;

		};

		struct text_box : public basic_text_box<fan_2d::rectangle_vector> {

			text_box(fan::camera& camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, const fan::color& text_color = fan::colors::white);

			void push_back(const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, const fan::color& text_color = fan::colors::white);

		};

		struct rounded_text_box : public basic_text_box<fan_2d::rounded_rectangle> {

			rounded_text_box(fan::camera& camera, const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, f_t radius, const fan::color& text_color = fan::colors::white);

			void push_back(const fan::fstring& text, f_t font_size, const fan::vec2& position, const fan::color& box_color, const fan::vec2& border_size, f_t radius, const fan::color& text_color = fan::colors::white);

		};

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

	class line_vector : public fan::basic_shape<true, fan::vec3>, public fan::basic_shape_color_vector<true> {
	public:

		line_vector(fan::camera& camera);
		line_vector(fan::camera& camera, const fan::mat2x3& begin_end, const fan::color& color);

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

		fan::window& m_window;

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

	class rectangle_vector : public fan::basic_shape<true, fan::vec3>, public fan::texture_handler {
	public:

		rectangle_vector(fan::camera& camera, const std::string& path, uint_t block_size);
		//rectangle_vector(fan::camera& camera, const fan::color& color, uint_t block_size);
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

		fan::window& m_window;

	private:
		fan::shader m_shader;

		fan::vec3 m_position;
		fan::vec3 m_size;

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

	#define d_grid_raycast_2d(start, end, raycast, block_size) \
		fan::grid_raycast_s<fan::vec2> raycast = { grid_direction(end, start), start, fan::vec2() }; \
		f_t _private_travel_distance = fan_2d::distance((start / block_size).floored(), (end / block_size).floored()); \
		if (!(start == end)) \
			while(grid_raycast_single(raycast, block_size) && _private_travel_distance >= fan_2d::distance((start / block_size).floored(), raycast.grid))

	#define d_grid_raycast_3d(start, end, raycast, block_size) \
		fan::grid_raycast_s<fan::vec3> raycast = { grid_direction(end, start), start, fan::vec3() }; \
		if (!(start == end)) \
			while(grid_raycast_single(raycast, block_size))
}