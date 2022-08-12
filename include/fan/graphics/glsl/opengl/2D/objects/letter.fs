R"(
#version 140

out vec4 color;

in vec4 text_color;
in vec2 texture_coordinate;
in float render_size;

uniform sampler2D _t00;

float outline_magic(float outline_size) {
  return 0.0f;
}

float get_outline_width(float outline_size) {
  return 0.5f;
}
float get_outline_edge(float outline_size) {
  return 0.1f;
}

void main() {
  float distance = texture(_t00, texture_coordinate).r;
  float smoothing = 1.0 / (render_size * 100);
  float width = 0.4;
  float alpha = smoothstep(width, width + smoothing, distance);

  float border_width = get_outline_width(0);
  float border_edge =  get_outline_edge(0);

  float outline_alpha = smoothstep(border_width, border_width + border_edge, distance);

 // vec3 final_color = mix(vec3(0, 0, 0), text_color.rgb, outline_alpha);

  if (alpha < 0.5) {
    discard;
  }
  color = vec4(text_color.rgb, alpha);

  //if (outline_alpha < 0.5) {
  //  color.rgb = vec3(0, 0, 0);
  //}
}
)"