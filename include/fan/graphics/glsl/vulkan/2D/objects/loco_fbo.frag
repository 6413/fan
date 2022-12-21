#version 420

layout(input_attachment_index=0, binding = 3) uniform subpassInput attachment0;
//layout(input_attachment_index=1, binding=5) uniform subpassInput attachment1;

layout(location = 0) out vec4 o_color;

void main() {
	o_color = subpassLoad(attachment0).r;
  o_color.a = 0;

}