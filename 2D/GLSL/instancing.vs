#version 330 core
layout (location = 0) in vec3 vertices;
layout (location = 1) in vec2 in_texture_coordinate;
layout (location = 3) in mat4 o_m;

out vec2 texture_coordinate;

uniform mat4 view;
uniform mat4 projection;

void main() {
     texture_coordinate = in_texture_coordinate;
     gl_Position = projection * view * o_m * vec4(vertices, 1.0);
}