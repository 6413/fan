#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec2 layout_position;
layout(location = 1) in vec4 layout_color;

layout(location = 0) out vec4 fragment_color;

void main() {
    gl_Position = ubo.proj * ubo.view * vec4(layout_position, 0.0, 1.0);
    fragment_color = layout_color;
}