#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec4 instance_outline_color;
layout(location = 2) in vec2 texture_coordinate;
layout(location = 3) in vec2 instance_size;
layout(location = 0) out vec4 o_attachment0;

float sd_box(vec2 p, vec2 half_size) {
  vec2 d = abs(p) - half_size;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

void main() {
  vec2 size = abs(instance_size);
  vec2 half_size = size * 0.5;
  vec2 p = (texture_coordinate - 0.5) * size * 2.0;
  float dist = sd_box(p, half_size);
  float outline_pixels = 1.0;
  float border = smoothstep(-outline_pixels - 0.5, -outline_pixels + 0.5, dist);
  float outer = smoothstep(-0.5, 0.5, dist);
  vec4 col = mix(instance_color, instance_outline_color, border);
  col.a *= 1.0 - outer;
  o_attachment0 = col;
}