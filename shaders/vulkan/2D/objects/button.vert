R"(
#version 450

#define get_instance() instances[gl_InstanceIndex]

struct out_data_t {
	vec4 color;
	vec4 outline_color;
	vec2 tc;
	float outline_size;
	float aspect_ratio;
};

layout(location = 0) out out_data_t out_data;

struct block_instance_t{
	vec3 position;
	float angle;
	vec2 size;
	vec2 rotation_point;
	vec4 color;
	vec4 outline_color;
	vec3 rotation_vector;
	float outline_size;
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
	vec2(-1, -1), // top left
	vec2(1, -1), // top right
	vec2(1, 1), // bottom right
	vec2(1, 1), // bottom right
	vec2(-1, 1), // bottom left
	vec2(-1, -1) // top left
);

void main() {
	uint id = uint(gl_VertexIndex % 6);

	vec2 rp = rectangle_vertices[id];
	
	float c = cos(-get_instance().angle);
	float s = sin(-get_instance().angle);

	float x = rp.x * c - rp.y * s;
	float y = rp.x * s + rp.y * c;

	mat4 view = pv[constants.camera_id].view;
  mat4 m = view;
	m[3][0] = 0;
	m[3][1] = 0;
	
	vec2 ratio_size = get_instance().size;

  gl_Position = pv[constants.camera_id].projection * m * vec4(vec2(x, y) * ratio_size + get_instance().position.xy + vec2(view[3][0], view[3][1]), get_instance().position.z, 1);

	out_data.color = get_instance().color;
	out_data.outline_color = get_instance().outline_color;
	out_data.tc = tc[id];
	out_data.outline_size = get_instance().outline_size;
	out_data.aspect_ratio = ratio_size.y / ratio_size.x;
}
)"