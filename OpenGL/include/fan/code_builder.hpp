private_template 
class private_class_name { 
public:

	static constexpr uint32_t gl_buffer =
		conditional_value<private_buffer_type == opengl_buffer_type::buffer_object, GL_ARRAY_BUFFER, 
		conditional_value<private_buffer_type == opengl_buffer_type::vertex_array_object, 0,
		conditional_value<private_buffer_type == opengl_buffer_type::shader_storage_buffer_object, GL_SHADER_STORAGE_BUFFER, static_cast<uint32_t>(fan::uninitialized)>::value>::value>::value;

	void allocate_buffer() {
		comparer<private_buffer_type>(
			[&] { glGenBuffers(1, &private_variable_name); },
			[&] { glGenVertexArrays(1, &private_variable_name); },
			[&] { glGenBuffers(1, &private_variable_name); },
			[&] { glGenTextures(1, &private_variable_name); }
		);
	}

	void free_buffer() {
		fan_validate_buffer(private_variable_name, {
			comparer<private_buffer_type>(
				[&] { glDeleteBuffers(1, &private_variable_name); },
				[&] { glDeleteVertexArrays(1, &private_variable_name); },
				[&] { glDeleteBuffers(1, &private_variable_name); },
				[&] { glDeleteTextures(1, &private_variable_name); }
			);
			private_variable_name = fan::uninitialized;
		});
	}

	private_class_name() : private_variable_name(fan::uninitialized) {
		this->allocate_buffer();
	}

	~private_class_name() {
		this->free_buffer();
	}

	
	private_class_name(const private_class_name& handler) : private_variable_name(fan::uninitialized) {
		this->allocate_buffer();
	}

	private_class_name(private_class_name&& handler) : private_variable_name(fan::uninitialized) {
		if ((int)handler.private_variable_name == fan::uninitialized) {
			throw std::runtime_error("attempting to move unallocated memory");
		}
		this->operator=(std::move(handler));
	}

	private_class_name& operator=(const private_class_name& handler) {

		this->free_buffer();

		this->allocate_buffer();

		return *this;
	}

	private_class_name& operator=(private_class_name&& handler) {

		this->free_buffer();

		this->private_variable_name = handler.private_variable_name;

		handler.private_variable_name = fan::uninitialized;

		return *this;
	}

	void bind_gl_storage_buffer(const std::function<void()> function) const {
		glBindBuffer(gl_buffer, private_variable_name);
		function();
		glBindBuffer(gl_buffer, 0);
	}

	template <opengl_buffer_type T = private_buffer_type, typename = std::enable_if_t<T == opengl_buffer_type::shader_storage_buffer_object>>
	void bind_gl_storage_buffer_base() const {
		glBindBufferBase(gl_buffer, private_layout_location, private_variable_name);
	}

	template <opengl_buffer_type T = private_buffer_type, typename = std::enable_if_t<T != opengl_buffer_type::texture && T != opengl_buffer_type::vertex_array_object>>
	void edit_data(void* data, uint_t offset, uint_t byte_size) {
		fan::edit_glbuffer(private_variable_name, data, offset, byte_size, gl_buffer, private_layout_location);
	}

	uint32_t private_variable_name;

protected:

	template <opengl_buffer_type T = private_buffer_type, typename = std::enable_if_t<T == opengl_buffer_type::buffer_object && T != opengl_buffer_type::vertex_array_object>>
	void edit_data(uint_t i, void* data, uint_t byte_size_single) {
		fan::edit_glbuffer(private_variable_name, data, i * byte_size_single, byte_size_single, gl_buffer, private_layout_location);
	}

	template <opengl_buffer_type T = private_buffer_type, typename = std::enable_if_t<T != opengl_buffer_type::texture && T != opengl_buffer_type::vertex_array_object>>
	void initialize_buffers(void* data, uint_t byte_size, bool divisor, uint_t attrib_count) {

		comparer<private_buffer_type>(

			[&] {
				glBindBuffer(gl_buffer, private_variable_name); 
			
				glEnableVertexAttribArray(private_layout_location);
				glVertexAttribPointer(private_layout_location, attrib_count, fan::GL_FLOAT_T, GL_FALSE, 0, 0);
			
				if (divisor) {
					glVertexAttribDivisor(private_layout_location, 1);
				}
			
				this->write_data(data, byte_size);
			},

			[] {}, 

			[&] {
				glBindBuffer(gl_buffer, private_variable_name); 
				glBindBufferBase(gl_buffer, private_layout_location, private_variable_name);
			
				if (divisor) {
					glVertexAttribDivisor(private_layout_location, 1);
				}
			
				this->write_data(data, byte_size);
			},

			[] {}
			
			); 
	}
	
	template <opengl_buffer_type T = private_buffer_type, typename = std::enable_if_t<T == opengl_buffer_type::vertex_array_object>>
	void initialize_buffers(uint32_t vao, const std::function<void()>& binder) {
		glBindVertexArray(vao);
		binder();
		glBindVertexArray(0);
	}

void write_data(void* data, uint_t byte_size) {
		fan::write_glbuffer(private_variable_name, data, byte_size, gl_buffer, private_layout_location); 
	}
};

#undef private_template
#undef private_class_name
#undef private_variable_name
#undef private_buffer_type
#undef private_layout_location