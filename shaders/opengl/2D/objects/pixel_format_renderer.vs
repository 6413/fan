
#version 330

out vec2 texture_coordinate;

uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_size;
layout (location = 2) in vec2 in_tc_position;
layout (location = 3) in vec2 in_tc_size;

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

  gl_Position = projection * view * vec4(rp * in_size + in_position.xy, in_position.z, 1);
  texture_coordinate = tc[id] * in_tc_size + in_tc_position;
}
