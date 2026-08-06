#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
  uint texture_id1;
  uint texture_id2;
  uint texture_id3;
  float time;
  uint lightmap_id;
  float ambient_floor;
  vec4 lighting_ambient;
} constants;

layout(set = 0, binding = 3) uniform u_t {
  vec3 u_top; // color 0.20 0.45 0.80
  vec3 u_bot; // color 0.60 0.85 0.95
} u;

void main() {
  float t = clamp(texture_coordinate.y, 0.0, 1.0);
  o_attachment0 = vec4(mix(u.u_bot, u.u_top, t) * instance_color.rgb, instance_color.a);
}
