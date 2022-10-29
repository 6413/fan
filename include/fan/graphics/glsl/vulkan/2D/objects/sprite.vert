#version 450

#define get_instance() instances[gl_InstanceIndex]

layout(location = 0) out vec4 instance_color;
layout(location = 1) out vec2 texture_coordinate;
layout(location = 2) out float instance_id;

struct block_instance_t{
	vec3 position;
	vec2 size;
	vec2 rotation_point;
	vec4 color;
	vec3 rotation_vector;
	float angle;
	vec2 tc_position;
	vec2 tc_size;
};

layout(std140, binding = 0) readonly buffer instances_t{
	block_instance_t instances[];
};

layout(binding = 1) uniform vp_t {
	mat4 projection;
  mat4 view;
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
	
	float c = cos(-get_instance().angle);
	float s = sin(-get_instance().angle);

	float x = rp.x * c - rp.y * s;
	float y = rp.x * s + rp.y * c;

	mat4 m = view;
	m[3][0] = 0;
	m[3][1] = 0;

  gl_Position = projection * m * vec4(vec2(x, y) * get_instance().size + get_instance().position.xy + vec2(view[3][0], view[3][1]), get_instance().position.z, 1);
	instance_color = get_instance().color;
	texture_coordinate = tc[id] * get_instance().tc_size + get_instance().tc_position;
	instance_id = gl_InstanceIndex / 204 + gl_InstanceIndex % 204;
	//texture_id = uint(tm[gl_InstanceIndex / 204 + gl_InstanceIndex % 204][0].x);
}