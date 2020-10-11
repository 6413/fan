#version 410 core

layout (location = 0) in vec4 color;
layout (location = 1) in vec3 vertices;
layout (location = 2) in vec2 texture_coordinates;
layout (location = 3) in vec3 normal;

uniform mat4 projection;
uniform mat4 view;

out vec4 t_color;

out vec2 texture_coordinate;

out vec3 normals;
out vec3 fragment_position;

void main() {
    normals = normal;
    texture_coordinate = texture_coordinates;
    t_color = color;
    gl_Position = projection * view * vec4(vertices, 1.0);
    fragment_position = gl_Position.xyz;
}