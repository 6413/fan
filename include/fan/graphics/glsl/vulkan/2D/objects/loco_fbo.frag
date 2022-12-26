R"(
#version 420

layout(input_attachment_index=0, binding = 3) uniform subpassInput attachment0;
layout(input_attachment_index=1, binding=4) uniform subpassInput attachment1;

layout(location = 0) out vec4 o_color;
//layout(location = 1) out vec4 r_color;

void main() {
  //o_color.r = 1;
	//o_color = subpassLoad(attachment0) + subpassLoad(attachment1);
  //o_color.a = 0;
  o_color = subpassLoad(attachment0) + subpassLoad(attachment1);
  vec4 c = o_color;

}
)"