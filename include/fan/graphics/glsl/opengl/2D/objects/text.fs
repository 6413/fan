R"(
#version 140

out vec4 color;

in vec4 text_color;
in vec2 texture_coordinate;
in float render_size;

uniform sampler2D texture_sampler;

float outline_magic(float outline_size) {
  return 0;
}

float get_outline_width(float outline_size) {
  return 0.5;
}
float get_outline_edge(float outline_size) {
  return 0.1;
}

void main() {
  float distance = texture(texture_sampler, texture_coordinate).r;
  float smoothing = 1.0 / (render_size * 100);
  float width = 0.4;
  float alpha = smoothstep(width, width + smoothing, distance);

  float border_width = get_outline_width(1);
  float border_edge =  get_outline_edge(1);

  float outline_alpha = smoothstep(border_width, border_width + border_edge, distance);

  vec3 final_color = mix(vec3(1, 0, 1), text_color.rgb, outline_alpha);

  color = vec4(final_color, outline_alpha);
}
)"