#version 330

in vec2 texture_coordinate;
layout (location = 0) out vec4 o_attachment0;

uniform sampler2D _t00; // color
uniform sampler2D _t01; // bloom
uniform float bloom_strength = 0.04;
uniform float gamma = 1.0;
uniform float exposure = 1.0;
uniform float contrast = 1.0;
uniform float framebuffer_alpha = 1.0;
uniform bool enable_bloom = true;

vec3 aces_tonemap(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec3 hdr_color = texture(_t00, texture_coordinate).rgb;
  vec3 color = hdr_color;

  if (enable_bloom) {
    vec3 bloom_color = texture(_t01, texture_coordinate).rgb;
    color += bloom_color * (bloom_strength * 10.0);
  }

  color = color * exposure;
  color = (color - 0.5) * contrast + 0.5;
  color = pow(max(color, vec3(0.0)), vec3(1.0 / gamma));

  o_attachment0 = vec4(color, 1.0);
}