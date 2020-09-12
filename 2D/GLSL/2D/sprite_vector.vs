#version 430 core
layout (location = 1) in vec2 position;
layout (location = 2) in vec2 size;

out vec2 texture_coordinate;

uniform mat4 projection;
uniform mat4 view;

const vec2 square_vertices[] = {
	vec2(0, 0),
	vec2(0, 1),
	vec2(1, 1),
	vec2(1, 1),
	vec2(1, 0),
	vec2(0, 0)
};

const vec2 texture_coordinates[] = {
	vec2( 1, 1),
	vec2( 0, 1),
	vec2( 0, 0),
	vec2( 0, 0),
	vec2( 0,-1),
	vec2(-1,-1)
};

out vec2 hori_blur_texture_coordinates[11];
out vec2 vert_blur_texture_coordinates[11];

void main() {
    texture_coordinate = texture_coordinates[gl_VertexID % texture_coordinates.length()];
	vec2 vertice = square_vertices[gl_VertexID % square_vertices.length()];
	gl_Position = projection * view * vec4(vec2(vertice.x * size.x + position.x, vertice.y * size.y + position.y), 0, 1);

	vec2 center_texture_coordinates = gl_Position.xy / 2 * 0.5 + 0.5;
	float pixel_size = 1.0f / 100.f;

	for (int i = -5; i < 5; i++) {
		hori_blur_texture_coordinates[i + 5] = texture_coordinate + vec2(pixel_size * i, 0);
		vert_blur_texture_coordinates[i + 5] = texture_coordinate + vec2(0, pixel_size * i);
	}
}