R"(
#version 330

layout (location = 0) out vec4 o_attachment0;
layout (location = 1) out uint o_attachment1;
layout (location = 2) out vec4 o_attachment2;

in vec2 texture_coordinate;
flat in uint flag;
in vec4 instance_color;

uniform sampler2D _t00;
uniform sampler2D _t02;
uniform vec3 lighting_ambient;

void main() {
  o_attachment0 = texture(_t00, texture_coordinate) * instance_color;
  vec4 t = vec4(texture(_t02, vec2(texture_coordinate.x, 1.0 - texture_coordinate.y)).rgb, 1);
  o_attachment0.rgb *= lighting_ambient;
  o_attachment2 = vec4(0, 0, 0, 0);
  //o_attachment2 = vec4(0, 0, 0, 0);
  //o_attachment1 = uint(texture(_t00, texture_coordinate).r * 255.0f) * 5u;
  //o_attachment1.g = 0u;
  //o_attachment1.b = 0;
  //o_attachment1.a = 255u;
}
)"