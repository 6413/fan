#version 330 core
in vec2 texture_coordinates;
out vec4 character_color;

in vec4 color;

uniform sampler2D text_texture;

void main() {
    character_color = color * vec4(1.0, 1.0, 1.0, texture(text_texture, texture_coordinates).r);
}