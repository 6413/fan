#version 130

out vec2 texture_coordinates;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

vec2 texture_coordinates_default[] = vec2[](
	vec2(1, 0),
	vec2(1, 1),
	vec2(0, 1),
	vec2(0, 0),
	vec2(-1, 0),
	vec2(-1, 1)
);

vec2 vertices[] = vec2[](
	vec2(1, 1),
	vec2(1, 0),
	vec2(0, 0),
	vec2(1, 1),
	vec2(0, 1),
	vec2(0, 0)
);

void main()
{
    gl_Position = projection * view * model * vec4(vertices[gl_VertexID % 6], 0, 1);
    texture_coordinates = vec2(
		texture_coordinates_default[gl_VertexID % 6].x, 
		1.0 - texture_coordinates_default[gl_VertexID % 6].y
	);
}