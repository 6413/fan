#version 430 core

layout (location = 4) in int characters;
layout (location = 6) in vec4 character_color;

out vec2 texture_coordinates;

uniform mat4 projection;

layout(std430, binding = 0) buffer vertex_layout 
{
    vec2 vertex[];
};

layout(std430, binding = 1) buffer texture_coordinate_layout 
{
    vec2 texture_coordinate[];
};

out vec4 text_color;

void main()
{
    gl_Position = projection * vec4(vertex[gl_VertexID + gl_InstanceID * 6], 0.0, 1.0);
    texture_coordinates = texture_coordinate[gl_VertexID + (characters - 33) * 6];
	text_color = character_color;
} 