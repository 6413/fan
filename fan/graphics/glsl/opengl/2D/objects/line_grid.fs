#version 330

layout (location = 0) out vec4 o_attachment0;

in vec4 instance_color;
in vec3 instance_position;
in vec3 frag_position;

in vec2 texture_coordinate;

uniform vec2 window_size;
uniform vec2 scaler;

void main() {
	o_attachment0 = vec4(texture_coordinate.x, texture_coordinate.y, 0, 1);
}