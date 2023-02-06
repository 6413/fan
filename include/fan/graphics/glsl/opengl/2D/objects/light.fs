R"(
#version 330

layout (location = 2) out vec4 o_attachment2;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;
in vec2 texture_coordinate;

const float gradient_depth = 600.0;

const vec3 u_sky_top_color = vec3(0.5, 0.8, 0.9) / 2;
const vec3 u_sky_bottom_color = vec3(0.5, 0.8, 0.9) / 30;

void main() {
  vec4 t = vec4(texture(_t02, texture_coordinate).rgb, 1);
  vec4 t2 = vec4(texture(_t00, texture_coordinate).rgb, 1);
  
  o_attachment2 = vec4(1, 1, 1, 1);
  //float distance = length(frag_position - instance_position);
  //float radius = instance_size.x;
  //float smooth_edge = radius;
  //float intensity = 1.0 - smoothstep(radius / 3 -smooth_edge, radius, distance);
  //o_attachment2 *= intensity;
}
)"