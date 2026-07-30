#version 450
layout(push_constant) uniform pc_t {
  vec4 color;
} pc;
layout(location = 0) out vec4 out_color;

float noise(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main() {
  float n = mix(0.96, 1.04, noise(gl_FragCoord.xy));
  out_color = vec4(pc.color.rgb * n, pc.color.a);
}
