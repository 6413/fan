#version 130

out vec4 color;

in vec2 texture_coordinates;

uniform sampler2D texture_sampler;
uniform float transparency;

void main()
{
    color = vec4(1, 1, 1, 1);
}  