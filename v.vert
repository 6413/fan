#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(binding = 1) uniform UniformBufferObject2 {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo2;


flat layout(location = 2) out uint id;

vec2 rectangle_vertices[] = vec2[](
	vec2(-1.0, -1.0),
	vec2(1.0, -1.0),
	vec2(1.0, 1.0),

	vec2(1.0, 1.0),
	vec2(-1.0, 1.0),
	vec2(-1.0, -1.0)
);

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(rectangle_vertices[gl_VertexIndex] * 0.3, 0, 1.0);
		id = gl_InstanceIndex;
}