R"(
#version 140

#define get_instance() instance.st[gl_VertexID / 6]

out vec4 text_color;
out vec2 texture_coordinate;
out float render_size;

uniform vec2 matrix_ratio;

layout (std140) uniform instance_t {
	struct{	
		vec2 position;
		vec2 size;
    vec4 color;
		vec2 tc_position;
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
  vec2(0, 1),
  vec2(1, 1),
  vec2(1, 0),
	vec2(1, 0),
	vec2(0, 0),
	vec2(0, 1)
);

vec2 swap(vec2 i) {
	return vec2(i.y, i.x);
}

void main() {
	uint id = uint(gl_VertexID % 6);

	vec2 ratio_size = get_instance().size * swap(matrix_ratio);

  gl_Position.xy = rectangle_vertices[id] * ratio_size + get_instance().position;
	text_color = get_instance().color;
	texture_coordinate = tc[id] * get_instance().tc_size + get_instance().tc_position;
	render_size = dot(get_instance().size, swap(matrix_ratio));
}
)"