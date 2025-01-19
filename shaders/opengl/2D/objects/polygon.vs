#version 330

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec4 in_color;

out vec4 color;

uniform mat4 view;
uniform mat4 projection;


void main() {
	gl_Position = projection * view * vec4(in_position, 1);
  color = in_color;
}