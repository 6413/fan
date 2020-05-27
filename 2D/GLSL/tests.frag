#version 330 core

in vec2 texture_coordinate;
out vec4 ShapeColor;

uniform sampler2D texture_sampler;

void main() {
    ShapeColor = texture2D(texture_sampler, texture_coordinate);
} 