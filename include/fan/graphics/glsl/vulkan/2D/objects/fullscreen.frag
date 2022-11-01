#version 450

layout(input_attachment_index=0, binding=4) uniform subpassInput attachment0;

layout(location = 0) out vec4 ocolor;

void main() {
	ocolor = subpassLoad(attachment0);
}