R"(
#version 330

layout (location = 0) out vec4 o_attachment0;
layout (location = 1) out uint o_attachment1;

in vec4 text_color;
in vec2 texture_coordinate;
in float render_size;

uniform sampler2D _t00;

void main() {
  o_attachment1 = 0u;
  float distance = texture(_t00, texture_coordinate).r;
  float smoothing = 1.0 / (log(render_size * 100) * 2);
  float width = 0.2;
  float alpha = smoothstep(width, width + smoothing, distance);

  float border_width = 0.5;
  float border_edge =  0.1;

  float outline_alpha = smoothstep(border_width, border_width + border_edge, distance);

 // vec3 final_color = mix(vec3(0, 0, 0), text_color.rgb, outline_alpha);

  o_attachment0 = vec4(text_color.rgb, alpha);
//  o_attachment0 = vec4(0.5, 0, 0, 1);
}
)"