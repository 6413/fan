#version 330 core
in vec2 texture_coordinates;

out vec4 sprite_color;

uniform sampler2D texture_sampler;
uniform float transparency;

void main()
{
    vec4 t = texture(texture_sampler, texture_coordinates);
    sprite_color = vec4(t.xyz, transparency);
    if (t.w != 1) {
        sprite_color.w = t.w * transparency;
    }
}