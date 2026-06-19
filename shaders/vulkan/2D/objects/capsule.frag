#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 instance_center0;
layout(location = 2) in vec2 instance_center1;
layout(location = 3) in float instance_radius;
layout(location = 4) in vec3 frag_position;
layout(location = 5) in vec4 instance_outline_color;
layout(location = 6) flat in uint flags;
layout(location = 0) out vec4 o_attachment0;

float sd_capsule(vec2 p, vec2 a, vec2 b, float r) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.000001), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

void main() {
  float dist = sd_capsule(frag_position.xy, instance_center0, instance_center1, instance_radius);
  float smoothing = max(fwidth(dist), 0.8);
  float alpha = 1.0 - smoothstep(-smoothing, smoothing, dist);
  float edge = -pow(instance_radius * 10.0, 1.0 / 3.0) / 3.0;
  float border = 1.0 - smoothstep(edge - smoothing, edge + smoothing, dist);
  vec4 color = vec4(mix(instance_outline_color.rgb, instance_color.rgb, border), alpha * instance_color.a);
  o_attachment0 = color;
}