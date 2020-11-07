#version 330 core
layout (location = 0) in vec3 position;
out vec3 texture_coordinates;

uniform mat4 projection;
uniform mat4 view;


void main()
{
    vec4 pos = projection * view * vec4(position, 1.0);
    gl_Position = pos.xyww;
    texture_coordinates = position;
}