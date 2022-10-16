#version 420

#define get_instance() instance[gl_VertexIndex / 6]

layout(location = 0) out vec4 instance_color;

struct block_instance_t{
	vec3 position;
	vec2 size;
	vec2 rotation_point;
	vec4 color;
	vec3 rotation_vector;
	float angle;
};

layout(binding = 0) uniform uniform_block_instance_t {
	block_instance_t instance[256];
};

layout(binding = 1) uniform vp_t {
  mat4 view;
  mat4 projection;
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
	uint id = uint( gl_VertexIndex % 6);

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