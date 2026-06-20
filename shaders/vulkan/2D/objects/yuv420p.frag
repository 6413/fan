#version 450

layout(location = 0) in vec2 texture_coordinate;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
  uint texture_id1;
  uint texture_id2;
  uint texture_id3;
} constants;

layout(set = 0, binding = 2) uniform sampler2D textures[1024];

void main() {
  float y = texture(textures[constants.texture_id], texture_coordinate).r;
  float u = texture(textures[constants.texture_id1], texture_coordinate).r;
  float v = texture(textures[constants.texture_id2], texture_coordinate).r;

  y = 1.1643 * (y - 0.0625);
  u -= 0.5;
  v -= 0.5;

  vec3 rgb;
  rgb.r = y + 1.5958 * v;
  rgb.g = y - 0.391773 * u - 0.81290 * v;
  rgb.b = y + 2.017 * u;

  o_attachment0 = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
