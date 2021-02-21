#version 130

attribute vec2 vertex;
attribute vec2 texture_coordinate;
attribute float font_sizes;
attribute vec4 text_colors;

out vec2 texture_coordinates;
out float font_size;
out vec3 text_color;

uniform mat4 projection;

void main() {
	gl_Position = projection * vec4(vertex, 0, 1);

    texture_coordinates = texture_coordinate;
    font_size = font_sizes;
    text_color = text_colors.xyz;
}