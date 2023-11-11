R"(
#version 450

#define get_instance() instances[gl_InstanceIndex]

layout(location = 0) out vec4 instance_color;
layout(location = 1) out vec2 texture_coordinate;
layout(location = 2) out float render_size;

struct block_instance_t{
	vec3 position;
  float outline_size;
	vec2 size;
	vec2 tc_position;
  vec4 color;
  vec4 outline_color;
	vec2 tc_size;
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
	uint id = uint(gl_VertexIndex % 6);

	vec2 rp = rectangle_vertices[id];
	
	mat4 view = pv[constants.camera_id].view;
	mat4 m = view;
	m[3][0] = 0;
	m[3][1] = 0;

  gl_Position = pv[constants.camera_id].projection * m * vec4(rectangle_vertices[id] * get_instance().size + get_instance().position.xy + vec2(view[3][0], view[3][1]), get_instance().position.z, 1);
	instance_color = get_instance().color;
	texture_coordinate = tc[id] * get_instance().tc_size + get_instance().tc_position;
	render_size = dot(get_instance().size, vec2(1, 1));
}
)"