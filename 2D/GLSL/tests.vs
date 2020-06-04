#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 in_texture_coordinate;

out vec2 texture_coordinate;

uniform mat4 view;
uniform mat4 projection;

void main() {
    vec4 l_position = projection * view * vec4(position, 1.0);
    texture_coordinate = in_texture_coordinate;
    gl_Position = l_position;
}