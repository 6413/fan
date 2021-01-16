#version 330 core

in vec2 texture_coordinate;
out vec4 shape_color;

uniform sampler2D texture_sampler;

void main() {
    shape_color = texture(texture_sampler, texture_coordinate);
}