#version 450

//layout(binding = 2) uniform sampler2D texSampler;

flat layout(location = 2) in uint id;

layout(location = 0) out vec4 outColor;

void main() {
	outColor = vec4(0, 0, 1, 1);
}