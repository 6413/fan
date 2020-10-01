#version 410 core

in vec4 t_color;

out vec4 shape_color;

void main() {
    shape_color = t_color;
} 