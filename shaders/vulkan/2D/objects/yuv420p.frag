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

float luma_at(vec2 tc) {
  return texture(textures[constants.texture_id], tc).r;
}

// three planes: u and v at half size (4:2:0) or full size (4:4:4)
vec2 chroma_at(vec2 tc) {
  return vec2(texture(textures[constants.texture_id1], tc).r, texture(textures[constants.texture_id2], tc).r);
}

// The colour of 4:2:0 video is one value per 2x2 pixels. Blending the four nearest by distance alone smeared it over
// the edges of letters and thin lines (coloured text looked blurred, a fringe on each side). The brightness is there
// for every pixel and shows the edge exactly: each of the four colours counts more the closer the brightness at its
// place is to this pixel's, so a letter's colour stays on the letter. 4:4:4 (colour at full size) is sampled as it is
vec2 chroma_for(vec2 tc, float y) {
  vec2 chroma_size = vec2(textureSize(textures[constants.texture_id1], 0));
  vec2 luma_size = vec2(textureSize(textures[constants.texture_id], 0));
  if (chroma_size.x * 1.5 > luma_size.x) {
    return chroma_at(tc);
  }
  vec2 position = tc * chroma_size - 0.5;
  vec2 base = floor(position);
  vec2 f = position - base;
  vec2 sum = vec2(0.0);
  float weights = 0.0;
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      vec2 at = (base + vec2(i, j) + 0.5) / chroma_size;
      float near = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
      // the brightness over this colour's 2x2 pixels (one bilinear sample at its center averages them)
      float difference = (luma_at(at) - y) * 14.0;
      float w = near / (1.0 + difference * difference) + 1e-4;
      sum += chroma_at(at) * w;
      weights += w;
    }
  }
  return sum / weights;
}

vec3 sample_rgb(vec2 tc) {
  float y = luma_at(tc);
  return yuv_to_rgb(y, chroma_for(tc, y));
}

// The picture drawn smaller than it is (a stream in a panel): one sample per screen pixel skips most of the
// picture's pixels, thin text breaks up and shimmers. Then four samples spread over the pixel's footprint are
// averaged. At its size or larger it is one sample
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
