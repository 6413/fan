#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec3 instance_position;
layout(location = 2) in vec2 instance_size;
layout(location = 3) in vec3 frag_position;
layout(location = 4) in vec2 uv;
layout(location = 5) flat in uint fs_flags;
layout(location = 0) out vec4 o_attachment0;

void main() {
  float intensity = 0.0;
  float radius = max(instance_size.x, 0.0001);
  float dist = length(frag_position.xy - instance_position.xy);
  if (fs_flags == 1u) {
    vec2 d = abs(frag_position.xy - instance_position.xy) / max(instance_size, vec2(0.0001));
    intensity = 1.0 - smoothstep(0.0, 1.0, max(d.x, d.y));
  }
  else {
    intensity = 1.0 - smoothstep(0.0, 1.0, length(uv));
  }
  intensity *= 1.0 - smoothstep(radius, radius * 1.2, dist);
  o_attachment0 = instance_color * max(intensity, 0.0);
}