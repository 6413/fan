R"(
#version 430

#define get_instance() instances[gl_InstanceIndex]

struct travel_data_t {
	vec4 color;
	float depth;
};

layout(location = 0) out travel_data_t data;

struct block_instance_t{
	vec3 position;
	vec2 size;
	vec2 rotation_point;
	vec4 color;
	vec3 rotation_vector;
	float angle;
};

layout(std140, binding = 0) readonly buffer instances_t{
	block_instance_t instances[];
};

layout(push_constant) uniform constants_t {
	uint texture_id;
	uint matrices_id;
}constants;

struct pv_t {
	mat4 projection;
	mat4 view;
};

layout(binding = 1) uniform upv_t {
	pv_t pv[16];
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
	uint id = uint(gl_VertexIndex % 6);

	vec2 rp = rectangle_vertices[id];
	
	float c = cos(-get_instance().angle);
	float s = sin(-get_instance().angle);

	float x = rp.x * c - rp.y * s;
	float y = rp.x * s + rp.y * c;

	mat4 view = pv[constants.matrices_id].view;
  mat4 m = view;
	m[3][0] = 0;
	m[3][1] = 0;
	
	vec4 view_position = m * vec4(vec2(x, y) * get_instance().size + get_instance().position.xy + vec2(view[3][0], view[3][1]), get_instance().position.z 1);

  gl_Position = pv[constants.matrices_id].projection * view_position;

	data.color = get_instance().color;
	data.depth = view_position.z;
}
)"