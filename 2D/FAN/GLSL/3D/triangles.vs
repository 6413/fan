#version 410 core

layout (location = 0) in vec4 color;
layout (location = 1) in vec3 vertices;

uniform mat4 projection;
uniform mat4 view;

out vec4 t_color;

void main() {
    gl_Position = projection * view * vec4(vertices, 1.0);
    t_color = color;
}