R"(
#version 140

#define get_instance() st[gl_VertexID / 6]

out vec4 instance_color;

uniform mat4 view;
uniform mat4 projection;

struct _{
	vec3 position;
	vec2 size;
	vec2 rotation_point;
	vec4 color;
	vec3 rotation_vector;
	float angle;
};

layout (std140) uniform instance_t {
	_ st[256];
};

vec2 rectangle_vertices[] = vec2[](
	vec2(-1.0, -1.0),
	vec2(1.0, -1.0),
	vec2(1.0, 1.0),

	vec2(1.0, 1.0),
	vec2(-1.0, 1.0),
	vec2(-1.0, -1.0)
);

void main() {
	uint id = uint(gl_VertexID % 6);

	vec2 ratio_size = get_instance().size;

	vec2 rp = rectangle_vertices[id];
	
	float c = cos(-get_instance().angle);
	float s = sin(-get_instance().angle);

	float x = rp.x * c - rp.y * s;
	float y = rp.x * s + rp.y * c;

  mat4 m = view;
	m[3][0] = 0;
	m[3][1] = 0;

  gl_Position = m * projection * vec4(vec2(x, y) * get_instance().size + get_instance().position.xy + vec2(view[3][0], view[3][1]), get_instance().position.z, 1);

	instance_color = get_instance().color;
}
)"