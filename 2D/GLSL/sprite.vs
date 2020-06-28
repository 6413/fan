#version 430 core

layout (location = 0) in vec2 vertex;
layout (location = 3) in mat4 model;

layout(std430, binding = 0) buffer texture_coordinate_layout 
{
    vec2 texture_coordinates[];
};

layout(std430, binding = 1) buffer textures_layout
{
    int textures[];
};

out vec2 texture_coordinate;
out vec2 blur_texture_coordinates[11];

uniform mat4 projection;
uniform mat4 view;

uniform float position_multiplier;

vec2 direction[360];

uniform bool vert;

void main() {
    for (float i = 0; i < 360.f; i++) {
        float theta = 2.0f * 3.1415926f * i / 360.f;
        float x = 1 * cos(theta);
        float y = 1 * sin(theta);
        direction[int(i)] = vec2(x, y);
    }
    int index = gl_VertexID + textures[gl_InstanceID] * 6;
    texture_coordinate = texture_coordinates[index];
    gl_Position = projection * view * model * vec4(vertex + direction[gl_InstanceID % 360] * position_multiplier, 0, 1.0);
    //vec2 center = vert ? texture_coordinate * 0.5 : texture_coordinate * 0.5 + 0.5;
    //float pixel_size = 1.0 / 1024;
    //for (int i = -5; i <= 5; i++) {
    //    blur_texture_coordinates[i + 5] = center + vec2(vert ? 0 : pixel_size * i, vert ? pixel_size * i : 0);
    //}
}