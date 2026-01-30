#version 330

layout (location = 0) out vec4 o_attachment0;

in vec4 instance_color;
in vec4 instance_outline_color;
in vec2 texture_coordinate;
in vec2 instance_size;

uniform float outline_pixels = 1.0;

float sd_box(vec2 p, vec2 half_size) {
  vec2 d = abs(p) - half_size;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

void main() {
  vec2 size = abs(instance_size);
  vec2 half_size = size * 0.5f;
  vec2 p = texture_coordinate * size - half_size;

  float dist = sd_box(p, half_size - outline_pixels);

  float border = dist > 0.0 ? 1.0 : 0.0;

  float outer = dist > outline_pixels ? 1.0 : 0.0;

  vec4 col = mix(instance_color, instance_outline_color, border);

  col.a *= 1.0 - outer;

  o_attachment0 = col;
}
