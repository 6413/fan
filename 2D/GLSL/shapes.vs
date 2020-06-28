#version 430 core
layout (location = 0) in vec2 vertex;
layout (location = 1) in vec4 in_color;
layout (location = 3) in mat4 model;

out vec4 color; 

uniform mat4 projection;
uniform mat4 view;

void main() {
    color = in_color;
    gl_Position = projection * view * model * vec4(vertex, 0, 1);
}