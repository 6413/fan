R"(
#version 140

#define get_instance() instance[gl_VertexID / 6]

out vec4 instance_color;
out vec2 instance_fragment_position;
out vec2 instance_position;
out float instance_radius;
out mat4 mvp;

uniform mat4 view;
uniform mat4 projection;

struct block_instance_t{
	vec3 position;
	float radius;
	vec2 rotation_point;
	vec4 color;
	vec3 rotation_vector;
	float angle;
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

void main() {
	uint id = uint(gl_VertexID % 6);

	vec2 rp = rectangle_vertices[id];
	
	float c = cos(-get_instance().angle);
	float s = sin(-get_instance().angle);

	float x = rp.x * c - rp.y * s;
	float y = rp.x * s + rp.y * c;

  mvp = projection * view;
  gl_Position = mvp * vec4(vec2(x, y) * vec2(get_instance().radius) + get_instance().position.xy, get_instance().position.z, 1);
	instance_color = get_instance().color;
  instance_radius = get_instance().radius;
  instance_position = get_instance().position.xy;
	instance_fragment_position = gl_Position.xy;
}
)"