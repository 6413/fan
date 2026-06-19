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
  yuv.x = texture(textures[constants.texture_id + 0], texture_coordinate).r - 0.0625;
  yuv.y = texture(textures[constants.texture_id + 1], texture_coordinate).r - 0.5;
  yuv.z = texture(textures[constants.texture_id + 1], texture_coordinate).g - 0.5;

  vec3 rgb;
  rgb.r = dot(yuv, vec3(1.164, 0.0, 1.596));
  rgb.g = dot(yuv, vec3(1.164, -0.391, -0.813));
  rgb.b = dot(yuv, vec3(1.164, 2.018, 0.0));

  o_attachment0 = vec4(rgb, 1);
}