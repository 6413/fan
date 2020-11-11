#version 430 core

layout(std430, binding = 0) buffer character_colors_layout 
{
    vec4 character_colors[];
};

layout (location = 1) in vec4 vertex; // <vec2 pos, vec2 tex>

out vec2 texture_coordinates;

out vec4 color;

uniform mat4 projection;

void main() {
    color = character_colors[gl_VertexID / 6];
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    texture_coordinates = vertex.zw;
}