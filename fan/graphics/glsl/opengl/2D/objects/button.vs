
  #version 140

  #define get_instance() instance[gl_VertexID / 6]

  out vec4 instance_color;
  out vec2 tcs;
	out vec4 outline_color;
	out float outline_size;
	out float aspect_ratio;

  uniform mat4 view;
  uniform mat4 projection;

	struct block_instance_t{
		vec3 position;
		vec2 size;
		vec2 rotation_point;
		vec3 angle;
		vec4 color;
		vec4 outline_color;
		float outline_size;
	};

  layout (std140) uniform instance_t {
		block_instance_t instance[204];
  };

  vec2 rectangle_vertices[] = vec2[](
	  vec2(-1.0, -1.0),
	  vec2(1.0, -1.0),
	  vec2(1.0, 1.0),

	  vec2(1.0, 1.0),
	  vec2(-1.0, 1.0),
	  vec2(-1.0, -1.0)
  );

  vec2 tc[] = vec2[](
	  vec2(-1, -1), // top left
	  vec2(1, -1), // top right
	  vec2(1, 1), // bottom right
	  vec2(1, 1), // bottom right
	  vec2(-1, 1), // bottom left
	  vec2(-1, -1) // top left
  );

  void main() {
	  uint id = uint(gl_VertexID % 6);

	  vec2 ratio_size = get_instance().size;

	  instance_color = get_instance().color;
    tcs = tc[id];
		aspect_ratio = ratio_size.y / ratio_size.x;
		outline_color = get_instance().outline_color;
		outline_size = get_instance().outline_size;

		vec2 rp = rectangle_vertices[id];
	
		float c = cos(0/*-get_instance().angle*/);
		float s = sin(0/*-get_instance().angle*/);

		float x = rp.x * c - rp.y * s;
		float y = rp.x * s + rp.y * c;

		gl_Position = projection * view * vec4(vec2(x, y) * ratio_size + get_instance().position.xy, get_instance().position.z, 1);
  }
