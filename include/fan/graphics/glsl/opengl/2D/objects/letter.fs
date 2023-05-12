R"(
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec4 text_color;
in vec4 outline_color;
in vec2 texture_coordinate;
in float render_size;

float outlineWidth = 0.01;

uniform sampler2D _t00;

void main() {

  //float distance = texture(_t00, texture_coordinate).r;
 
  //// Convert glyphAdvance from NDC to texture space
  // float glyphAdvance = render_size * textureSize(_t00, 0).x;

  // // Calculate the size of the glyph being rendered
  // vec2 glyphSize = vec2(glyphAdvance, 1.0) / textureSize(_t00, 0);

  //vec2 outlineWidthTexel = vec2(outlineWidth) / glyphSize;

  //// Calculate the distance from the edge of the glyph
  //float d = distance - 0.5;

  //vec3 blendedColor = mix(text_color.rgb, outline_color.rgb,  smoothstep(0.0, outlineWidthTexel.x, abs(d)));
  // float opacity = (outline_color.a) / (text_color.a - outline_color.a);
  //o_attachment0 = vec4(blendedColor.rgb, 1 - opacity);


  float distance = texture(_t00, texture_coordinate).r;
  float smoothing = 1.0 / (render_size * 100 * 2);
  float width = 0.1;
  float alpha = smoothstep(width, width + smoothing, distance);

  float border_width = 0.1;
  float border_edge =  0.5;

  float outline_alpha = smoothstep(border_width, border_width + border_edge, distance);

  vec3 final_color = mix(outline_color.rgb, text_color.rgb, outline_alpha);

  o_attachment0 = vec4(final_color.rgb, alpha - 1 + text_color.a + 0.3);
}
)"