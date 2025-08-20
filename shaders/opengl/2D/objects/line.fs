#version 330 core

in vec4 instance_color;
in vec2 line_coord;

out vec4 color;

void main() {
  float distance_from_center = abs(line_coord.y);
  float alpha = 1.0 - smoothstep(0.5, 1.0, distance_from_center);
  color = vec4(instance_color.rgb, instance_color.a * alpha);
}