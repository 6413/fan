R"(
#version 330

layout (location = 0) out vec4 o_attachment0;
layout (location = 1) out uint o_attachment1;
layout (location = 2) out vec4 o_attachment2;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;
in vec2 texture_coordinate;

void main() {
vec4 t2 = vec4(texture(_t00, texture_coordinate).rgb, 1);
  vec4 t = vec4(texture(_t02, texture_coordinate).rgb, 1);

  o_attachment2 = instance_color;
  //o_attachment2.r = 1;
  //o_attachment2 = mix(instance_color, t, 0.5);

  vec3 lightDir = normalize(instance_position - frag_position);
  float distance = length(frag_position - instance_position);
  float radius = instance_size.x;
  float smooth_edge = 0.1;
  float intensity = 1.0 - smoothstep(radius / 3 -smooth_edge, radius, distance);
  o_attachment2 *= intensity;
  //diffuse  *= intensity;
  //specular *= intensity;

}
)"