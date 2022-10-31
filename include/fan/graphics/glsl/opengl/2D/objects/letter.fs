R"(
#version 140

out vec4 color;

in vec4 text_color;
in vec2 texture_coordinate;
in float render_size;

uniform sampler2D _t00;

void main() {
  float distance = texture(_t00, texture_coordinate).r;
  float smoothing = 1.0 / (render_size * 40);
  float width = 0.2;
  float alpha = smoothstep(width, width + smoothing, distance);

  float border_width = 0.5;
  float border_edge =  0.1;

  float outline_alpha = smoothstep(border_width, border_width + border_edge, distance);

 // vec3 final_color = mix(vec3(0, 0, 0), text_color.rgb, outline_alpha);

  color = vec4(text_color.rgb, alpha);

//  color = vec4(0.5, 0, 0, 1);
}
)"