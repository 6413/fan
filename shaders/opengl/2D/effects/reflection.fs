#version 330
in vec2 texture_coordinate;
out vec4 o_color;
uniform sampler2D _t00;
uniform float water_y;
uniform float water_height;
uniform float alpha;
uniform float distortion_strength;
uniform float distortion_speed;
uniform float distortion_scale;
uniform float distortion_octaves;
uniform float distortion_lacunarity;
uniform float distortion_gain;
uniform float wave_amplitude;
uniform float wave_frequency;
uniform float wave_speed;
uniform float caustic_strength;
uniform float caustic_speed;
uniform float caustic_scale;
uniform float highlight_strength;
uniform float highlight_power;
uniform float shimmer_strength;
uniform float depth_fade_power;
uniform float tint_strength;
uniform vec4 shallow_color;
uniform vec4 deep_color;
uniform float _time;

vec2 hash22(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(dot(hash22(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
        dot(hash22(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
    mix(dot(hash22(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
        dot(hash22(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x),
    u.y
  );
}

float fbm(vec2 p, int octaves, float lacunarity, float gain) {
  float value = 0.0;
  float amplitude = 0.5;
  for (int i = 0; i < octaves; i++) {
    value += amplitude * noise(p);
    p *= lacunarity;
    amplitude *= gain;
  }
  return value;
}

void main() {
  float water_top = (0.5 - water_y);
  float water_bottom = (0.5 - water_y) + water_height;
  if (texture_coordinate.y < water_top || texture_coordinate.y > water_bottom) {
    discard;
  }

  vec2 tc = texture_coordinate;
  tc.y = 1.0 - tc.y;

  float relative_y = (texture_coordinate.y - water_top) / water_height;
  relative_y = clamp(relative_y, 0.0, 1.0);

  float freeze = smoothstep(0.08, 0.25, relative_y);

  vec2 distortion_uv = texture_coordinate * distortion_scale;
  distortion_uv.y -= _time * distortion_speed;

  int oct = int(distortion_octaves);
  float dx = fbm(distortion_uv, oct, distortion_lacunarity, distortion_gain);
  float dy = fbm(distortion_uv + vec2(5.2, 1.3), oct, distortion_lacunarity, distortion_gain);
  float wave = sin(texture_coordinate.y * wave_frequency + _time * wave_speed) * wave_amplitude;

  vec2 distortion_offset = vec2(dx + wave, dy) * 0.5;
  float depth_curve = pow(relative_y, depth_fade_power);
  distortion_offset *= distortion_strength * depth_curve * freeze;
  distortion_offset.y = clamp(distortion_offset.y, 0.0, 1.0);

  tc += distortion_offset;

  float horizon_v = 1.0 - water_top;
  float clamp_buffer = 0.003;
  float max_valid_y = horizon_v - clamp_buffer;
  if (tc.y > max_valid_y) {
    tc.y = max_valid_y;
  }

  tc.x = clamp(tc.x, 0.0, 1.0);
  tc.y = clamp(tc.y, 0.0, max_valid_y);

  vec3 color = texture(_t00, tc).rgb;

  vec4 tint = mix(shallow_color, deep_color, relative_y * tint_strength);
  color *= tint.rgb;

  float highlight = (dx + dy + 2.0) * 0.25;
  highlight = pow(highlight, highlight_power) * highlight_strength * (1.0 - pow(relative_y, 0.8));
  color += vec3(highlight);

  vec2 cuv = distortion_uv * caustic_scale + _time * caustic_speed;
  float caustic = fbm(cuv, 3, 2.0, 0.5);
  caustic = caustic * caustic_strength + (1.0 - caustic_strength * 0.5);
  color *= mix(1.0, caustic, relative_y * 0.5);

  float shimmer = pow(1.0 - relative_y, 2.0) * shimmer_strength;
  color += vec3(shimmer);

  color *= mix(1.0, 0.8, relative_y * 0.6);

  float depth_alpha = alpha * mix(0.85, 0.92, 1.0 - relative_y * 0.3);
  o_color = vec4(color, depth_alpha * tint.a);
}