#version 330 core

in vec2 texture_coordinates;

out vec4 color;

uniform sampler2D texture_sampler;

void main() {
    color = texture(texture_sampler, texture_coordinates);
}