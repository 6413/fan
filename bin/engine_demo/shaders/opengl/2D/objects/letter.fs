#version 330

layout (location = 0) out vec4 o_attachment0;

in vec4 text_color;
in vec4 outline_color;
in vec2 texture_coordinate;
in float render_size;
in float outline_size;

uniform sampler2D _t00;

void main() {
  float distance = texture(_t00, texture_coordinate).r;
  float outlineWidth = outline_size / (render_size * 1000 * 2);
  float smoothing = 1.0 / (render_size * 1000 * 2);
  float outerEdgeCenter = 0.5 - outlineWidth;
  float alpha = smoothstep(outlineWidth - smoothing, outerEdgeCenter + smoothing, distance);

  float border = smoothstep(0.5 - smoothing, 0.5 + smoothing, distance);

  o_attachment0 = vec4(mix(outline_color.rgb, text_color.rgb, border), alpha);
}