R"(
#version 140

#define get_instance() instance[gl_VertexID / 6]

out vec4 text_color;
out vec4 outline_color;
out vec2 texture_coordinate;
out float render_size;

uniform mat4 view;
uniform mat4 projection;

struct block_instance_t{	
	vec3 position;
  float outline_size;
	vec2 size;
	vec2 tc_position;
  vec4 color;
  vec4 outline_color;
	vec2 tc_size;
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
	vec2(0, 0), // top left
	vec2(1, 0), // top right
	vec2(1, 1), // bottom right
	vec2(1, 1), // bottom right
	vec2(0, 1), // bottom left
	vec2(0, 0) // top left
);

void main() {
	uint id = uint(gl_VertexID % 6);

  gl_Position = projection * view * vec4(rectangle_vertices[id] * get_instance().size + get_instance().position.xy, get_instance().position.z, 1);

	text_color = get_instance().color;
  outline_color = get_instance().outline_color;
	texture_coordinate = tc[id] * get_instance().tc_size + get_instance().tc_position;
	render_size = get_instance().size.y;
}
)"