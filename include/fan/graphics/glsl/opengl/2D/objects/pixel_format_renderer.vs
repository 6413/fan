R"(
#version 140

#define get_instance() instance[gl_VertexID / 6]

out vec2 texture_coordinate;

uniform mat4 view;
uniform mat4 projection;

struct block_instance_t{
	vec3 position;
	vec2 size;
	vec2 tc_position;
	vec2 tc_size;
};

layout (std140) uniform instance_t {
	block_instance_t instance[256];
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
	//
  vec2 rp = rectangle_vertices[id];

  gl_Position = projection * view * vec4(rp * get_instance().size + get_instance().position.xy, get_instance().position.z, 1);
	texture_coordinate = tc[id] * get_instance().tc_size + get_instance().tc_position;
}
)"