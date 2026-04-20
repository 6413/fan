#version 330
in vec2 texture_coordinate;
layout (location = 0) out vec4 o_attachment0;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;

uniform float bloom_strength = 0.04;
uniform float bloom_intensity = 1.0;
uniform vec3 bloom_tint = vec3(1.0, 1.0, 1.0);
uniform float dirt_intensity = 1.0;

uniform float gamma = 1.0;
uniform float exposure = 1.0;
uniform float contrast = 1.0;
uniform float framebuffer_alpha = 1.0;
uniform bool enable_bloom = true;

void main() {
  vec3 color = texture(_t00, texture_coordinate).rgb;

  if (enable_bloom) {
    vec3 bloom = texture(_t01, texture_coordinate).rgb;
    
    bloom = bloom * bloom_tint * bloom_intensity * (bloom_strength * 10.0);

    if (dirt_intensity > 0.0) {
        vec3 dirt = texture(_t02, texture_coordinate).rgb;
        bloom += bloom * dirt * dirt_intensity;
    }

    color += bloom;
  }

  color *= exposure;
  color = (color - 0.5) * contrast + 0.5;
  // already srgb
  //color = pow(max(color, vec3(0.0)), vec3(1.0 / gamma));

  o_attachment0 = vec4(color, framebuffer_alpha);
}