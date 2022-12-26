R"(
#version 330

layout (location = 0) out vec4 o_attachment0;
layout (location = 1) out uint o_attachment1;

in vec4 instance_color;

float rand(vec2 co){
return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  //if (instance_color.a < 0.9) {
  //  discard;
  //}
  o_attachment0 = instance_color;
}
)"