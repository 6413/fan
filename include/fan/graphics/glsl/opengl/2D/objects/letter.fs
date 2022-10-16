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
  float smoothing = 1.0 / (render_size * 40);
  float width = 0.2;
  float alpha = smoothstep(width, width + smoothing, distance);

  float border_width = get_outline_width(0);
  float border_edge =  get_outline_edge(0);

  float outline_alpha = smoothstep(border_width, border_width + border_edge, distance);

 // vec3 final_color = mix(vec3(0, 0, 0), text_color.rgb, outline_alpha);

  if (alpha < 0.9) {
    //color = vec4(vec3(0.3, 0, 0), alpha);
    discard;
  }
  else {
    
    color = vec4(text_color.rgb, alpha);
  }

//  color = vec4(0.5, 0, 0, 1);
}
)"