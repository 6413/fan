#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 line_start;
layout(location = 2) in vec2 line_end;
layout(location = 3) in float line_radius;
layout(location = 4) in vec3 frag_position;
layout(location = 5) in vec2 ba;
layout(location = 6) in float ba_len2;
layout(location = 0) out vec4 o_attachment0;

float sd_capsule(vec2 p, vec2 a, float r) {
  vec2 pa = p - a;
  float h = clamp(dot(pa, ba) / ba_len2, 0.0, 1.0);
  return length(pa - ba * h) - r;
}

void main() {
  float d = sd_capsule(frag_position.xy, line_start, line_radius);
  float smoothing = max(fwidth(d), 0.5);
  float alpha = 1.0 - smoothstep(-smoothing, smoothing, d);
  o_attachment0 = vec4(instance_color.rgb, alpha * instance_color.a);
}