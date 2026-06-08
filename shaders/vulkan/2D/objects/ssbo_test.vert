#version 430

//layout(location = 0) out vec4 instance_color;

vec2 rectangle_vertices[] = vec2[](
	vec2(-1.0, -1.0),
	vec2(1.0, -1.0),
	vec2(1.0, 1.0),

	vec2(1.0, 1.0),
	vec2(-1.0, 1.0),
	vec2(-1.0, -1.0)
);

struct test_struct_t {
	mat4 m;
};

layout(std140, binding = 0) readonly buffer instances_t{
	test_struct_t instances[];
};

void main() {

	mat4 p = mat4(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 1
	);

	float rectangle_scale = 0.3;
	
	vec2 ssbodata = instances[gl_InstanceIndex].m[0].xy;

  gl_Position = p * vec4(rectangle_vertices[gl_VertexIndex] * rectangle_scale + ssbodata, 0, 1);

	//instance_color = get_instance().color;
}