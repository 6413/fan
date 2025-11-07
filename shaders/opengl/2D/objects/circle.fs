#version 330

layout (location = 0) out vec4 o_attachment0;

in vec4 instance_color;
in vec3 instance_position;
in float instance_radius;
in vec3 frag_position;
in vec2 texture_coordinate;
flat in uint flags;

uniform float camera_zoom;

out vec4 color;

void main() {
  float distance = length(frag_position - instance_position);
  float radius = instance_radius;

  float smooth_edge = 2.0 / camera_zoom;

  float intensity = clamp(1.0 - smoothstep(radius - smooth_edge, radius, distance), 0.0, 1.0);
  vec3 base_color = instance_color.rgb;
  vec4 color = vec4(base_color, instance_color.a * intensity);

  o_attachment0 = color;
}
