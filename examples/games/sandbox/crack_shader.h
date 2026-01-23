inline static const char* crack_shader_fragment = R"(#version 330
layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;

uniform sampler2D _t00;
uniform float crack_strength;

float hash(float p){
  p = fract(p * 443.897);
  p += p * 19.19;
  return fract(p * p);
}

float crack_segment(vec2 uv, vec2 a, vec2 b, vec2 c, float thickness, float progress, float seed){
  float min_d = 10.0;
  float step = 0.006;

  for (float t = 0.0; t <= progress; t += step){
    float s = 1.0 - t;

    vec2 p = s*s*a + 2.0*s*t*b + t*t*c;

    float jitter = (hash(t * 123.45 + seed * 91.7) - 0.5) * 0.02;
    p += vec2(jitter, jitter);

    float taper = mix(1.0, 0.3, t);
    float d = length(uv - p) / taper;
    min_d = min(min_d, d);

    if (hash(t * 99.1 + seed * 13.7) > 0.94){
      vec2 jdir = normalize(vec2(
        hash(t * 7.1 + seed) - 0.5,
        hash(t * 5.3 + seed) - 0.5
      ));
      float d2 = length(uv - (p + jdir * 0.04));
      min_d = min(min_d, d2);
    }
  }

  return smoothstep(thickness, 0.0, min_d);
}

void main(){
  vec2 uv = texture_coordinate;

  float seed = instance_color.a;
  float progress = crack_strength;

  vec2 center = vec2(0.5) +
    (vec2(hash(seed * 12.3), hash(seed * 17.9)) - 0.5) * 0.12;

  float m = 0.0;

  const int NUM_CRACKS = 100;

  for (int i = 0; i < NUM_CRACKS; i++){
    float s = seed * (10.0 + float(i) * 3.17);

    float angle = hash(s) * 6.28318;
    angle += (hash(s * 8.1) - 0.5) * 0.4;

    float length = 0.35 + hash(s * 1.3) * 0.55;
    float curve = (hash(s * 2.1) - 0.5) * 0.35;

    vec2 dir = vec2(cos(angle), sin(angle));
    vec2 endp = center + dir * length;

    vec2 perp = vec2(-dir.y, dir.x);
    vec2 ctrl = mix(center, endp, 0.5) + perp * curve;

    float thickness = 0.010 + hash(s * 3.7) * 0.004;

    m = max(m, crack_segment(uv, center, ctrl, endp, thickness, progress, seed));

    if (progress > 0.45 && hash(s * 4.9) > 0.6){
      vec2 bstart = mix(center, endp, 0.5 + hash(s * 5.3) * 0.3);
      float bang = angle + (hash(s * 6.1) - 0.5) * 1.2;
      vec2 bdir = vec2(cos(bang), sin(bang));
      vec2 bend = bstart + bdir * 0.25;
      vec2 bctrl = mix(bstart, bend, 0.5) + vec2(-bdir.y, bdir.x) * 0.1;

      float bp = (progress - 0.45) * 2.0;
      m = max(m, crack_segment(uv, bstart, bctrl, bend, thickness * 0.7, bp, seed));
    }
  }

  m = clamp(m, 0.0, 1.0);

  float r = length(uv - center);
  float shock = smoothstep(0.6, 0.2, r) * 0.1 * progress;

  vec3 crack_color = vec3(0.0);

  vec3 final_rgb = crack_color * m;
  float final_a = m;

  final_rgb *= 1.0 - shock;

  o_attachment0 = vec4(final_rgb, final_a);
})";
