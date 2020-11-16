#version 430 core
layout (location = 1) in vec2 position;
layout (location = 2) in vec2 size;

out vec2 texture_coordinate;

uniform mat4 projection;
uniform mat4 view;

const vec2 square_vertices[] = {
	vec2(1, 0),
	vec2(0, 0),
	vec2(0, 1),
	vec2(0, 1),
	vec2(1, 1),
	vec2(1, 0)
};

const vec2 texture_coordinates[] = {
	vec2(1, 0),
	vec2(0, 0),
	vec2(0, 1),
	vec2(0, 1),
	vec2(1, 1),
	vec2(1, 0)
};

void main() {
    texture_coordinate = texture_coordinates[gl_VertexID % texture_coordinates.length()];
	vec2 vertice = square_vertices[gl_VertexID % square_vertices.length()];
	gl_Position = projection * view * vec4(vec2(vertice.x * size.x + position.x, vertice.y * size.y + position.y), 0, 1);
}