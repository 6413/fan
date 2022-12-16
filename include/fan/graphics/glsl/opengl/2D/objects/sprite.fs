R"(
#version 330

layout (location = 0) out vec4 o_attachment0;
//layout (location = 1) out uvec4 o_attachment1;

in vec2 texture_coordinate;

uniform sampler2D _t00;

void main() {
  o_attachment0 = texture(_t00, texture_coordinate);
  //o_attachment1.r = uint(texture(_t00, texture_coordinate).r * 255.0f);
  //o_attachment1.g = 0;
  //o_attachment1.b = 0;
  //o_attachment1.a = 255u;a
}
)"