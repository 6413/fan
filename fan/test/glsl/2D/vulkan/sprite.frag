#version 450

layout(location = 0) in vec2 fragment_texture_coordinate;

layout(location = 0) out vec4 color;

layout(binding = 1) uniform sampler2D texture_sampler;

void main() {
    color = texture(texture_sampler, vec2(fragment_texture_coordinate.x, 1.0 - fragment_texture_coordinate.y));
}