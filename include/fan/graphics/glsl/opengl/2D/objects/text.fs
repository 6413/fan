R"(
#version 140

out vec4 color;

in vec4 text_color;
in vec2 texture_coordinate;
in float render_size;

uniform sampler2D texture_sampler;

float outline_magic(float outline_size) {
  return outline_size / 7 + 0.065;
}

float get_outline_width(float outline_size) {
  return 0.5 + outline_magic(outline_size);
}
float get_outline_edge(float outline_size) {
  return 0.1 + outline_magic(outline_size);
}

void main() {
  float distance = texture(texture_sampler, texture_coordinate).r;
  float smoothing = 1.0 / (render_size * 30);
  float width = 0.4;
  float alpha = smoothstep(width, width + smoothing, distance);

  color = vec4(text_color.rgb, alpha);
}
)"