#version 430 core

out vec2 texture_coordinates;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

const vec2 texture_coordinates_default[] = {
	vec2( 1, 1),
	vec2( 0, 1),
	vec2( 0, 0),
	vec2( 0, 0),
	vec2( 0,-1),
	vec2(-1,-1)
};

const vec2 vertices[] = {
	vec2(0.5, -0.5),
	vec2(-0.5, -0.5),
	vec2(-0.5, 0.5),
	vec2(0.5, -0.5),
	vec2(0.5, 0.5),
	vec2(-0.5, 0.5)
};

void main()
{
    gl_Position = projection * view * model * vec4(vertices[gl_VertexID % vertices.length()], 0, 1);
    texture_coordinates = vec2(
		texture_coordinates_default[gl_VertexID % texture_coordinates_default.length()].x, 
		1.0 - texture_coordinates_default[gl_VertexID % texture_coordinates_default.length()].y
	);
}