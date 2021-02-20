#version 130

varying vec2 texture_coordinates;

out vec4 sprite_color;

uniform sampler2D texture_sampler;
uniform float transparency;

void main()
{
    sprite_color = texture(texture_sampler, texture_coordinates);
}