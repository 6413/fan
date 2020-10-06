#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 layout_normal;
layout (location = 2) in vec2 layout_texture_coordinates;

out vec2 texture_coordinates;
out vec3 normal;
out vec3 fragment_position;
out float visibility;

const float density = 0.05;
const float gradient = 1.5;

uniform mat4 view;
uniform mat4 projection;

void main() {
    vec4 world_position = vec4(position, 1);
    vec4 relative = view * world_position;
    gl_Position = vec4(projection * relative).xyww;
    float distance = length(relative.xyz);
    visibility = exp(-pow((distance * density), gradient));
    visibility = clamp(visibility, 0, 1);

    texture_coordinates = layout_texture_coordinates;
}