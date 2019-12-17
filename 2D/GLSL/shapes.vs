#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 inColor;
out vec4 color;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    color = inColor;
}