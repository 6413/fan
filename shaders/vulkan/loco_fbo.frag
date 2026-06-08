#version 420

layout(input_attachment_index = 0, set = 0, binding = 1) uniform subpassInput attachment0;

//layout(location = 0) in vec4 in_color;
layout(location = 0) out vec4 o_color;
//layout(location = 1) out vec4 r_color;

void main() {
  o_color = subpassLoad(attachment0);
}