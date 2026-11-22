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

// video is display encoded, the scene is linear and the swapchain encodes it again
vec3 srgb_to_linear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

vec3 yuv_to_rgb(float y, vec2 uv) {
  y = 1.1643 * (y - 0.0625);
  uv -= vec2(0.5);
  vec3 rgb;
  rgb.r = y + 1.5958 * uv.y;
  rgb.g = y - 0.391773 * uv.x - 0.81290 * uv.y;
  rgb.b = y + 2.017 * uv.x;
  return clamp(rgb, 0.0, 1.0);
}

// three planes: 4:2:0 (u and v at half size) and 4:4:4 (all at full size) alike, each sampled at the same place
vec3 sample_rgb(vec2 tc) {
  float y = texture(textures[constants.texture_id], tc).r;
  float u = texture(textures[constants.texture_id1], tc).r;
  float v = texture(textures[constants.texture_id2], tc).r;
  return yuv_to_rgb(y, vec2(u, v));
}

// The picture drawn smaller than it is (a stream in a panel): one bilinear sample per screen pixel skips most of the
// picture's pixels, thin text breaks up and shimmers. Then four samples spread over the pixel's footprint are
// averaged (16 picture pixels). At its size or larger it is one sample, as before
void main() {
  vec2 step_uv = fwidth(texture_coordinate);
  vec2 texels = step_uv * vec2(textureSize(textures[constants.texture_id], 0));
  vec3 rgb;
  if (max(texels.x, texels.y) > 1.05) {
    vec2 d = step_uv * 0.25;
    rgb = (sample_rgb(texture_coordinate + vec2(-d.x, -d.y)) + sample_rgb(texture_coordinate + vec2(d.x, -d.y)) +
      sample_rgb(texture_coordinate + vec2(-d.x, d.y)) + sample_rgb(texture_coordinate + vec2(d.x, d.y))) * 0.25;
  }
  else {
    rgb = sample_rgb(texture_coordinate);
  }
  o_attachment0 = vec4(srgb_to_linear(rgb), 1.0);
}
