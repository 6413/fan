#version 450

layout(location = 0) in vec2 v_texcoords;
layout(location = 1) flat in vec4 v_color0;
layout(location = 2) flat in vec4 v_color1;
layout(location = 3) flat in vec4 v_color2;
layout(location = 4) flat in vec4 v_color3;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
  uint texture_id1;
  uint texture_id2;
  uint texture_id3;
  uint pad0;
  uint pad1;
  float ambient_floor;
  vec4 lighting_ambient;
} constants;

vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }

float dither(vec2 p) {
  return fract(sin(dot(p, vec2(0.9898, 5.233))) * 43758.5453) - 0.5;
}

void main() {
  vec3 c0 = srgb_to_linear(v_color0.rgb);
  vec3 c1 = srgb_to_linear(v_color1.rgb);
  vec3 c2 = srgb_to_linear(v_color2.rgb);
  vec3 c3 = srgb_to_linear(v_color3.rgb);

  vec3 color_left = mix(c0, c3, v_texcoords.y);
  vec3 color_right = mix(c1, c2, v_texcoords.y);
  vec3 result = mix(color_left, color_right, v_texcoords.x);

  float alpha = mix(mix(v_color0.a, v_color3.a, v_texcoords.y),
                    mix(v_color1.a, v_color2.a, v_texcoords.y),
                    v_texcoords.x);

  result += dither(gl_FragCoord.xy) / 128.0;
  o_attachment0 = vec4(result, alpha);
}