R"(
#version 140

#define get_instance() instance.st[gl_VertexID / 6]

out vec4 text_color;
out vec2 texture_coordinate;
out float render_size;

uniform mat4 view;
uniform mat4 projection;

layout (std140) uniform instance_t {
	struct{	
		vec3 position;
    float outline_size;
		vec2 size;
		vec2 tc_position;
    vec4 color;
    vec4 outline_color;
		vec2 tc_size;
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

vec2 swap(vec2 i) {
	return vec2(i.y, i.x);
}

void main() {
	uint id = uint(gl_VertexID % 6);

	vec2 swapped = vec2(1, 1);
	vec2 ratio_size = get_instance().size * swapped;

	mat4 m = view;
	m[3][0] = 0;
	m[3][1] = 0;

  gl_Position = m * projection * vec4(rectangle_vertices[id] * ratio_size + get_instance().position.xy + vec2(view[3][0], view[3][1]), get_instance().position.z, 1);

	text_color = get_instance().color;
	texture_coordinate = tc[id] * get_instance().tc_size + get_instance().tc_position;
	render_size = dot(get_instance().size, swapped);
}
)"