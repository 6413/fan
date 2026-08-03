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
   float time;
   float shadow_map_scale;
 } pc;
layout(location = 0) out vec4 out_color;
const float tau = 6.28318530718;

float noise(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123);
}

float linstep(float lo, float hi, float v) {
  return clamp((v - lo) / (hi - lo), 0.0, 1.0);
}

float shadow_vsm(vec2 moments, float dist) {
  float p = step(dist, moments.x);
  float variance = max(moments.y - moments.x * moments.x, 0.0001);
  float d = dist - moments.x;
  float p_max = linstep(0.2, 1.0, variance / (variance + d * d));
  return max(p, p_max);
}

vec2 shadow_sample(float angle) {
  return texture(shadow_texture, vec2(fract(angle) * pc.shadow_map_scale, 0.5)).rg;
}

void main() {
  vec2 p = v_uv * 2.0 - 1.0;
  float dist = length(p);
  if (dist > 1.0) discard;
  float fu = fract((atan(p.y, p.x) / tau) + 1.0);

  float step_a = pc.angle_texel * pc.softness;
  float s1 = step_a, s2 = step_a * 2.0, s3 = step_a * 3.0, s4 = step_a * 4.0;
  float s5 = step_a * 5.0, s6 = step_a * 6.0, s7 = step_a * 7.0, s8 = step_a * 8.0;

  vec2 moments = vec2(0.0);
  moments += shadow_sample(fu - s8) * 0.004;
  moments += shadow_sample(fu - s7) * 0.008;
  moments += shadow_sample(fu - s6) * 0.016;
  moments += shadow_sample(fu - s5) * 0.027;
  moments += shadow_sample(fu - s4) * 0.045;
  moments += shadow_sample(fu - s3) * 0.063;
  moments += shadow_sample(fu - s2) * 0.094;
  moments += shadow_sample(fu - s1) * 0.117;
  moments += shadow_sample(fu)       * 0.252;
  moments += shadow_sample(fu + s1) * 0.117;
  moments += shadow_sample(fu + s2) * 0.094;
  moments += shadow_sample(fu + s3) * 0.063;
  moments += shadow_sample(fu + s4) * 0.045;
  moments += shadow_sample(fu + s5) * 0.027;
  moments += shadow_sample(fu + s6) * 0.016;
  moments += shadow_sample(fu + s7) * 0.008;
  moments += shadow_sample(fu + s8) * 0.004;

  float lit = shadow_vsm(moments, dist);

  float pixel_angle = atan(p.y, p.x);
  float diff = abs(mod(pixel_angle - pc.cone_angle + 3.14159, tau) - 3.14159);
  float cone_mask = 1.0 - smoothstep(pc.cone_inner * 0.5, pc.cone_outer * 0.5, diff);

  float falloff = pow(max(1.0 - dist, 0.0), pc.falloff_power);
  falloff *= mix(0.97, 1.03, noise(gl_FragCoord.xy));
  float flicker = 0.95 + 0.05 * sin(pc.time * 8.0 + noise(p * 10.0) * 6.28);

  float value = lit * falloff * cone_mask * flicker;
  vec3 light_col = mix(pc.light_color.rgb, pc.light_color.rgb * vec3(0.6, 0.7, 1.0), dist);

  out_color = vec4(light_col * value * 2.0 * pc.light_color.a, 0.0);
}
