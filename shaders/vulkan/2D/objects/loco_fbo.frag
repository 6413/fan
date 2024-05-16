R"(
#version 420

layout(input_attachment_index=0, binding = 3) uniform subpassInput attachment0;
layout(input_attachment_index=1, binding = 4) uniform usubpassInput attachment1;

//layout(location = 0) in vec4 in_color;
layout(location = 0) out vec4 o_color;
//layout(location = 1) out vec4 r_color;

void main() {
  o_color = vec4(subpassLoad(attachment0).rgb, 1);

}
)"