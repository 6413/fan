#version 450

layout(location = 0) in vec2 local_position;
layout(location = 1) in vec2 grid_size;
layout(location = 2) in vec4 instance_color;
layout(location = 0) out vec4 o_attachment0;

void main() {
  vec2 line_dist = abs(fract(local_position / grid_size - 0.5) - 0.5) * grid_size;
  vec2 aa = max(fwidth(local_position), vec2(0.5));
  vec2 intensity = 1.0 - smoothstep(aa, aa * 2.0, line_dist);
  float line = max(intensity.x, intensity.y);
  o_attachment0 = instance_color * line;
}