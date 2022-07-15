R"(
  #version 140

  #define get_instance() instance.st[gl_VertexID / 6]

  out vec4 instance_color;
  out vec2 tcs;
	out vec4 outline_color;
	out float outline_size;
	out float aspect_ratio;

  uniform mat4 view;
  uniform mat4 projection;

  layout (std140) uniform instance_t {
	  struct{
		  vec2 position;
		  vec2 size;
		  vec4 color;
		  vec3 rotation_vector;
		  float angle;
			vec4 outline_color;
		  vec2 rotation_point;
			float outline_size;
	  }st[256];
  }instance;

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
	
		float c = cos(get_instance().angle);
		float s = sin(get_instance().angle);

		float x = rp.x * c - rp.y * s;
		float y = rp.x * s + rp.y * c;

		gl_Position = view * projection * vec4(vec2(x, y) * ratio_size + get_instance().position, 0, 1);
  }
)"