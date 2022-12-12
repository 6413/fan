R"(
#version 330

layout (location = 0) out vec3 o_fbo_out;
layout (location = 1) out uint o_bit_flag;

in vec2 texture_coordinate;

in vec4 instance_color;

out vec4 o_color;

uniform sampler2D _t00;

void main() {
  o_color = texture(_t00, texture_coordinate);
  //o_fbo_out.r *= 1.000001;
  o_bit_flag = 255u;
  //if (o_color.a < 0.9) {
  //  discard;
  //}
}
)"