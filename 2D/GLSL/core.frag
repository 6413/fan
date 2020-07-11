#version 330 core
in vec2 TexCoord;

out vec4 Color;

uniform sampler2D texture_sampler;


void main()
{
    Color = texture(texture_sampler, TexCoord);
}