#version 130

attribute vec4 in_color;
attribute vec2 position;
attribute vec2 size;

varying vec4 color;

uniform mat4 projection;
uniform mat4 view;

void main() {
    color = in_color;
	gl_Position = projection * view * vec4(position.x, position.y, 0, 1);
}