R"(
#version 150

#define get_instance() instance.st[gl_VertexID / 6]

out vec4 instance_color;
out vec2 texture_coordinate;

uniform mat4 view;
uniform mat4 projection;

struct _{
	vec2 position;
	vec2 size;
	vec4 color;
	vec3 rotation_vector;
	float angle;
	vec2 rotation_point;
	vec2 tc_position;
	vec2 tc_size;
};

layout (std140) uniform instance_t {
	_ st[128];
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
	vec2(0, 0), // top left
	vec2(1, 0), // top right
	vec2(1, 1), // bottom right
	vec2(1, 1), // bottom right
	vec2(0, 1), // bottom left
	vec2(0, 0) // top left
);

void main() {
	uint id = uint(gl_VertexID % 6);

	vec2 rp = rectangle_vertices[id];
	
	float c = cos(-get_instance().angle);
	float s = sin(-get_instance().angle);

	float x = rp.x * c - rp.y * s;
	float y = rp.x * s + rp.y * c;

	mat4 m = view;
	m[3][0] = 0;
	m[3][1] = 0;

  gl_Position = m * projection * vec4(vec2(x, y) * get_instance().size + get_instance().position + vec2(view[3][0], view[3][1]), 0, 1);
	instance_color = get_instance().color;
	texture_coordinate = tc[id] * get_instance().tc_size + get_instance().tc_position;
}
)"