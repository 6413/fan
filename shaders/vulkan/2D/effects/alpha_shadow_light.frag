#version 450
layout(location = 0) in vec2 v_uv;
layout(set = 0, binding = 0) uniform sampler2D shadow_texture;
layout(push_constant) uniform pc_t {
  vec2 ndc_min;
  vec2 ndc_max;
  vec4 light_color;
  float softness;
  float falloff_power;
  float angle_texel;
  float cone_angle;
  float cone_inner;
  float cone_outer;
} pc;
layout(location = 0) out vec4 out_color;
const float tau = 6.28318530718;

float sample_shadow(float u, float dist) {
  float blocker = texture(shadow_texture, vec2(fract(u), 0.5)).r;
  float penumbra = pc.softness * (1.0 - blocker) + 0.001;
  return 1.0 - smoothstep(max(blocker - penumbra, blocker * 0.95), blocker + penumbra, dist);
}

void main() {
  vec2 p = v_uv * 2.0 - 1.0;
  float dist = length(p);
  if (dist > 1.0) discard;
  float u = atan(p.y, p.x) / tau;
  if (u < 0.0) u += 1.0;
  float lit = 0.0;
  lit += sample_shadow(u - pc.angle_texel * 4.0, dist) * 0.02;
  lit += sample_shadow(u - pc.angle_texel * 3.0, dist) * 0.06;
  lit += sample_shadow(u - pc.angle_texel * 2.0, dist) * 0.12;
  lit += sample_shadow(u - pc.angle_texel,       dist) * 0.20;
  lit += sample_shadow(u,                     dist) * 0.20;
  lit += sample_shadow(u + pc.angle_texel,       dist) * 0.20;
  lit += sample_shadow(u + pc.angle_texel * 2.0, dist) * 0.12;
  lit += sample_shadow(u + pc.angle_texel * 3.0, dist) * 0.06;
  lit += sample_shadow(u + pc.angle_texel * 4.0, dist) * 0.02;
  float pixel_angle = atan(p.y, p.x);
  float diff = abs(mod(pixel_angle - pc.cone_angle + 3.14159, tau) - 3.14159);
  float cone_mask = 1.0 - smoothstep(pc.cone_inner * 0.5, pc.cone_outer * 0.5, diff);
  float falloff = pow(max(1.0 - dist, 0.0), pc.falloff_power);
  float value = lit * falloff * cone_mask;
  out_color = vec4(pc.light_color.rgb * value, pc.light_color.a * value);
}
