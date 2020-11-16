#version 430 core
layout (location = 1) in vec2 vertex;
layout (location = 2) in vec2 texture_coordinate;

out vec2 texture_coordinates;
out float font_size;
out vec3 outline_color;
out vec3 text_color;

uniform mat4 projection;
uniform mat4 view;

layout(std430, binding = 0) buffer text_color_layout
{
    vec4 text_colors[];
};

layout(std430, binding = 3) buffer letter_edges_layout
{
    float font_sizes[];
};

layout(std430, binding = 4) buffer outline_color_layout
{
    vec4 outline_colors[];
};

void main() {
    font_size = font_sizes[gl_VertexID / 6];
    outline_color = outline_colors[gl_VertexID / 6].xyz;
    text_color = text_colors[gl_VertexID / 6].xyz;
    texture_coordinates = texture_coordinate;
	gl_Position = projection * view * vec4(vertex, 0, 1);
}