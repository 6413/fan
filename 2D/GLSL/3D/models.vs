#version 330 core

layout (location = 3) in vec3 vertex;
layout (location = 4) in vec3 normal;
layout (location = 5) in vec2 texture_coordinate;

out vec2 texture_coordinates;
out vec3 normals;
out vec3 fragment_position;
out float visibility;

const float density = 0.05;
const float gradient = 1.5;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;

void main() {
    vec4 world_position = model * vec4(vertex, 1);
    vec4 relative = view * world_position;

    gl_Position = projection * relative;

    //fragment_position = vec3(model * vec4(position, 1.0f));
    //normals = mat3(transpose(inverse(model))) * normal;
    //float distance = length(relative.xyz);
    //visibility = exp(-pow((distance * density), gradient));
    //visibility = clamp(visibility, 0, 1);

    texture_coordinates = texture_coordinate;
}