#version 330 core
in vec2 v_uv;
uniform sampler2D shadow_texture;
uniform vec4 light_color;
uniform float softness;
uniform float falloff_power;
uniform float angle_texel;
out vec4 out_color;
float sample_shadow(float u, float dist) {
  float blocker = texture(shadow_texture, vec2(fract(u), 0.5)).r;
  float penumbra = softness * (1.0 - blocker) + 0.001;
  return 1.0 - smoothstep(max(blocker - penumbra, blocker * 0.95), blocker + penumbra, dist);
}
void main() {
  const float tau = 6.28318530718;
  vec2 p = v_uv * 2.0 - 1.0;
  float dist = length(p);
  if (dist > 1.0) discard;
  float u = atan(p.y, p.x) / tau;
  if (u < 0.0) u += 1.0;
  float lit = 0.0;
  lit += sample_shadow(u - angle_texel * 4.0, dist) * 0.02;
  lit += sample_shadow(u - angle_texel * 3.0, dist) * 0.06;
  lit += sample_shadow(u - angle_texel * 2.0, dist) * 0.12;
  lit += sample_shadow(u - angle_texel,       dist) * 0.20;
  lit += sample_shadow(u,                     dist) * 0.20;
  lit += sample_shadow(u + angle_texel,       dist) * 0.20;
  lit += sample_shadow(u + angle_texel * 2.0, dist) * 0.12;
  lit += sample_shadow(u + angle_texel * 3.0, dist) * 0.06;
  lit += sample_shadow(u + angle_texel * 4.0, dist) * 0.02;
  float falloff = pow(max(1.0 - dist, 0.0), falloff_power);
  out_color = vec4(light_color.rgb * lit * falloff, light_color.a * lit * falloff);
}
