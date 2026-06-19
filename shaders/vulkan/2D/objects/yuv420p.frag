#version 450

layout(location = 0) in vec2 texture_coordinate;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
} constants;

layout(set = 0, binding = 2) uniform sampler2D textures[1024];

void main() {
  vec3 yuv;
  yuv.x = texture(textures[constants.texture_id + 0], texture_coordinate).r;
  yuv.y = texture(textures[constants.texture_id + 1], texture_coordinate).r;
  yuv.z = texture(textures[constants.texture_id + 2], texture_coordinate).r;

  yuv.x = 1.1643 * (yuv.x - 0.0625);
  yuv.y -= 0.5;
  yuv.z -= 0.5;

  vec3 rgb;
  rgb.r = yuv.x + 1.5958 * yuv.z;
  rgb.g = yuv.x - 0.391773 * yuv.y - 0.81290 * yuv.z;
  rgb.b = yuv.x + 2.017 * yuv.y;

  o_attachment0 = vec4(rgb, 1);
}