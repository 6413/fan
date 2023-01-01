R"(
#version 330

layout (location = 0) out vec4 o_attachment0;
layout (location = 1) out uint o_attachment1;

in vec2 texture_coordinate;
in flat uint flag;
in vec4 instance_color;

uniform sampler2D _t00;

void main() {
  o_attachment0 = texture(_t00, texture_coordinate) * instance_color;
  //o_attachment1 = uint(texture(_t00, texture_coordinate).r * 255.0f) * 5u;
  //o_attachment1.g = 0u;
  //o_attachment1.b = 0;
  //o_attachment1.a = 255u;
}
)"