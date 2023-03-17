R"(
#version 330

layout (location = 1) out vec4 o_attachment1;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;

const float gradient_depth = 600.0;

const vec3 u_sky_top_color = vec3(0.5, 0.8, 0.9) / 2;
const vec3 u_sky_bottom_color = vec3(0.5, 0.8, 0.9) / 30;

void main() {
  o_attachment1 = instance_color;
  float distance = length(frag_position - instance_position);
  float radius = instance_size.x;
  float smooth_edge = radius;
  float intensity = 1.0 - smoothstep(radius / 3 -smooth_edge, radius, distance);
  o_attachment1 *= intensity;
}
/*
o_attachment1 = instance_color;
float distance = length(frag_position - instance_position);
float radius = instance_size.x;
float smooth_edge = radius;
float intensity = clamp((radius - distance + radius / 3) / (2 * radius / 3), 0.0, 1.0);
o_attachment1 *= intensity;
*/

)"