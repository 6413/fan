#version 130

precision highp float;

out vec4 shape_color;
in vec2 texture_coordinate;

in vec4 color;

uniform bool enable_texture;
uniform sampler2D texture_sampler;

void main()
{
    if (enable_texture) {
        shape_color = texture(texture_sampler, texture_coordinate);
    }
    else {
        shape_color = color;
    }

} 