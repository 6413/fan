#version 410 core

in vec4 t_color;

in vec2 texture_coordinate;

out vec4 shape_color;

uniform sampler2D texture1;

void main() {
    shape_color = t_color;
} 