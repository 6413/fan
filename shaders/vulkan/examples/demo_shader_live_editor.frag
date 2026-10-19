#version 450
layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
  uint texture_id1;
  uint texture_id2;
  uint texture_id3;
} constants;

layout(set = 0, binding = 2) uniform sampler2D textures[1024];

void DrawVignette(inout vec3 color, vec2 uv) {
  float vignette = uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y);
  vignette = clamp(pow(16.0 * vignette, 0.3), 0.0, 1.0);
  color *= vignette;
}

vec2 CRTCurveUV(vec2 uv) {
  uv = uv * 2.0 - 1.0;
  vec2 offset = abs(uv.yx) / vec2(6.0, 4.0);
  uv = uv + uv * offset * offset;
  uv = uv * 0.5 + 0.5;
  return uv;
}

void DrawScanline(inout vec3 color, vec2 uv) {
  float scanline = clamp(0.95 + 0.05 * cos(3.14 * uv.y * 240.0), 0.0, 1.0);
  float grille = 0.85 + 0.15 * clamp(1.5 * cos(3.14 * uv.x * 640.0), 0.0, 1.0);
  color *= scanline * grille * 1.2;
}

void main() {
  vec2 tex = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);
  tex = CRTCurveUV(tex * 1.05);
  vec3 actual = texture(textures[constants.texture_id], tex).rgb;
  o_attachment0 = vec4(actual * instance_color.rgb, instance_color.a);
  DrawVignette(o_attachment0.rgb, tex);
  DrawScanline(o_attachment0.rgb, tex);
}