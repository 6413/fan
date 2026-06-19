#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec3 instance_position;
layout(location = 2) in float instance_radius;
layout(location = 3) in vec3 frag_position;
layout(location = 4) flat in uint flags;
layout(location = 5) in vec4 instance_outline_color;
layout(location = 6) in float instance_outline_width;
layout(location = 0) out vec4 o_attachment0;

void main() {
  float dist = length(frag_position.xy - instance_position.xy);
  float smoothing = max(fwidth(dist), 0.5);
  float outer_alpha = 1.0 - smoothstep(instance_radius - smoothing, instance_radius, dist);
  float inner = instance_radius - instance_outline_width;
  float inner_alpha = 1.0 - smoothstep(inner - smoothing, inner, dist);
  vec4 result = mix(instance_outline_color, instance_color, inner_alpha);
  result.a *= outer_alpha;
  o_attachment0 = result;
}