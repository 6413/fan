#version 330 core

out vec3 texture_coordinates;

uniform mat4 projection;
uniform mat4 view;

vec3 skybox_vertices[36] = (vec3[36])(
	vec3(-1.0,  1.0, -1.0),
	vec3(-1.0, -1.0, -1.0),
	vec3(1.0, -1.0, -1.0),
	vec3(1.0, -1.0, -1.0),
	vec3(1.0,  1.0, -1.0),
	vec3(-1.0,  1.0, -1.0),

	vec3(-1.0, -1.0,  1.0),
	vec3(-1.0, -1.0, -1.0),
	vec3(-1.0,  1.0, -1.0),
	vec3(-1.0,  1.0, -1.0),
	vec3(-1.0,  1.0,  1.0),
	vec3(-1.0, -1.0,  1.0),

	vec3(1.0, -1.0, -1.0),
	vec3(1.0, -1.0,  1.0),
	vec3(1.0,  1.0,  1.0),
	vec3(1.0,  1.0,  1.0),
	vec3(1.0,  1.0, -1.0),
	vec3(1.0, -1.0, -1.0),

	vec3(-1.0, -1.0,  1.0),
	vec3(-1.0,  1.0,  1.0),
	vec3(1.0,  1.0,  1.0),
	vec3(1.0,  1.0,  1.0),
	vec3(1.0, -1.0,  1.0),
	vec3(-1.0, -1.0,  1.0),

	vec3(-1.0,  1.0, -1.0),
	vec3(1.0,  1.0, -1.0),
	vec3(1.0,  1.0,  1.0),
	vec3(1.0,  1.0,  1.0),
	vec3(-1.0,  1.0,  1.0),
	vec3(-1.0,  1.0, -1.0),

	vec3(-1.0, -1.0, -1.0),
	vec3(-1.0, -1.0,  1.0),
	vec3(1.0, -1.0, -1.0),
	vec3(1.0, -1.0, -1.0),
	vec3(-1.0, -1.0,  1.0),
	vec3(1.0, -1.0,  1.0)
);

void main()
{
    vec4 pos = projection * view * vec4(position, 1.0);
    gl_Position = pos.xyww;
    texture_coordinates = position;
}