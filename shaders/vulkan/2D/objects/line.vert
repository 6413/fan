R"(
#version 430

#define get_instance() instances[gl_InstanceIndex]

struct travel_data_t {
	vec4 color;
};

layout(location = 0) out travel_data_t data;

struct block_instance_t {
	vec4 color;
	vec3 src;
	vec3 dst;
};

layout(std140, binding = 0) readonly buffer instances_t{
	block_instance_t instances[];
};

layout(push_constant) uniform constants_t {
	uint texture_id;
	uint camera_id;
}constants;

struct pv_t {
	mat4 projection;
	mat4 view;
};

layout(binding = 1) uniform upv_t {
	pv_t pv[16];
};

void main() {
	uint id = uint(gl_VertexIndex % 6);

	mat4 view = pv[constants.camera_id].view;
  mat4 m = view;
	m[3][0] = 0;
	m[3][1] = 0;
	
	vec4 view_position = m * vec4(((id & 1u) == 0u) ? get_instance().src.xy : get_instance().dst.xy, get_instance().src.z, 1);

  gl_Position = pv[constants.camera_id].projection * view_position;

	data.color = get_instance().color;
}
)"