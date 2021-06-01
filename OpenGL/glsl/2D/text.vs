#version 120

attribute vec2 vertex;
attribute vec2 texture_coordinate;
attribute float font_sizes;
attribute vec4 text_colors;

varying vec2 texture_coordinates;
varying float font_size;
varying vec3 text_color;

uniform mat4 projection;

void main() {
    texture_coordinates = texture_coordinate;
    text_color = text_colors.xyz;
    font_size = font_sizes;
	gl_Position = projection * vec4(vertex + vec2(0.5, 0.5), 0, 1);
}