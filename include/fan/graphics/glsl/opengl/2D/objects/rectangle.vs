R"(
#version 140

#define get_instance() instance.st[gl_VertexID / 6]

out vec4 instance_color;

uniform mat4 view;
uniform mat4 projection;

layout (std140) uniform instance_t {
	struct{
		vec2 position;
		vec2 size;
		vec4 color;
		float angle;
		vec2 rotation_point;
		vec3 rotation_vector;
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
	vec2(0, 0), // top left
	vec2(1, 0), // top right
	vec2(1, 1), // bottom right
	vec2(1, 1), // bottom right
	vec2(0, 1), // bottom left
	vec2(0, 0) // top left
);

void main() {
	uint id = uint(gl_VertexID % 6);

	vec2 ratio_size = get_instance().size;

  gl_Position = view * projection * vec4(rectangle_vertices[id] * ratio_size + get_instance().position, 0, 1);
	instance_color = get_instance().color;
}
)"