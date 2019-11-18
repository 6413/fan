#version 330 compatibility
layout (location = 0) in vec4 vertex; // <vec2 position, vec2 texCoords>

out vec2 TexCoords;
out vec4 ParticleColor;

uniform mat4 projection;
uniform vec2 offset;
uniform vec4 color;
uniform mat4 view;

void main()
{
    float scale = 1000.0f;
    TexCoords = vertex.zw;
    ParticleColor = color;
    gl_Position = projection * view * vec4((vertex.xy * scale) + offset, 0.0, 1.0);
}