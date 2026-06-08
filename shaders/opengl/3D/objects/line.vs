#version 330

out vec4 instance_color;

uniform mat4 view;
uniform mat4 projection;

layout (location = 0) in vec4 color;
layout (location = 1) in vec3 src;
layout (location = 2) in vec3 dst;

void main() {
	uint id = uint(gl_VertexID);

	gl_Position = projection * view * vec4(((id & 1u) == 0u) ? src : dst, 1);
	instance_color = color;
}
