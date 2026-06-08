#version 330
layout (location = 0) out vec4 o_attachment0;
in vec2 texture_coordinate;

uniform float _time;
uniform vec2 window_size;
uniform vec4 custom_color = vec4(188.0/255.0, 164.0/255.0, 164.0/255.0, 1.0);

uniform float u_pixel_size = 1200.0f;
uniform float u_scale_sparse = 4.5f;
uniform float u_cov_sparse = 0.50f;
uniform float u_warp_sparse = 2.1f;
uniform float u_scale_dense = 3.8f;
uniform float u_cov_dense = 2.95f;
uniform float u_warp_dense = 3.7f;
uniform float u_dense_height = 0.45f;
uniform float u_dense_puffiness = 4.3f;
uniform float u_dense_variation = 0.4f;

uniform float u_density = 4.7f;
uniform float u_shadow_str = 2.5f;
uniform float u_light_x = 0.0f;
uniform float u_light_y = 1.0f;

uniform vec3 u_col_sky_top = vec3(0.10f, 0.29f, 0.47f);
uniform vec3 u_col_sky_bot = vec3(0.42f, 0.69f, 0.90f);
uniform vec3 u_col_cloud = vec3(1.00f, 1.00f, 1.00f);
uniform vec3 u_col_shadow = vec3(0.48f, 0.58f, 0.66f);

const mat2 fbm_mat = mat2(1.6, 1.2, -1.2, 1.6);
const vec2 hash_c1 = vec2(127.1, 311.7);
const vec2 hash_c2 = vec2(269.5, 183.3);

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  
  vec2 i00 = i;
  vec2 i10 = i + vec2(1.0, 0.0);
  vec2 i01 = i + vec2(0.0, 1.0);
  vec2 i11 = i + vec2(1.0, 1.0);
  
  vec2 h00 = fract(sin(vec2(dot(i00, hash_c1), dot(i00, hash_c2))) * 43758.5453123) * 2.0 - 1.0;
  vec2 h10 = fract(sin(vec2(dot(i10, hash_c1), dot(i10, hash_c2))) * 43758.5453123) * 2.0 - 1.0;
  vec2 h01 = fract(sin(vec2(dot(i01, hash_c1), dot(i01, hash_c2))) * 43758.5453123) * 2.0 - 1.0;
  vec2 h11 = fract(sin(vec2(dot(i11, hash_c1), dot(i11, hash_c2))) * 43758.5453123) * 2.0 - 1.0;
  
  return mix(
    mix(dot(h00, f), dot(h10, f - vec2(1.0, 0.0)), u.x),
    mix(dot(h01, f - vec2(0.0, 1.0)), dot(h11, f - vec2(1.0, 1.0)), u.x),
    u.y
  ) * 0.5 + 0.5;
}

float fbm(vec2 p) {
  float f = 0.0, amp = 0.5;
  for (int i = 0; i < 4; i++) {
    f += amp * noise(p);
    p = fbm_mat * p;
    amp *= 0.5;
  }
  return f;
}

float map_sparse(vec2 p, float coverage) {
  vec2 q = vec2(fbm(p), fbm(p + vec2(5.2, 1.3)));
  return smoothstep(1.0 - coverage, 1.0, fbm(p + q * u_warp_sparse));
}

float map_dense(vec2 p, float coverage) {
  vec2 q = vec2(fbm(p), fbm(p + vec2(5.2, 1.3)));
  float d = fbm(p + q * u_warp_dense);
  float n = smoothstep(1.0 - coverage, 1.0, d);
  return pow(n, u_dense_puffiness);
}
/*
  float get_clouds(vec2 uv, vec2 base_p) {
  float d1 = 0.0;
  float clump_mask = smoothstep(0.2, 0.7, fbm(base_p * 2.0 + _time * 0.002));
  
  if (clump_mask > 0.001) {
  vec2 p1 = base_p * u_scale_sparse * vec2(1.0, 3.0);
  p1.x += _time * 0.005;
  d1 = map_sparse(p1, u_cov_sparse) * clump_mask * 0.7;
  }
  
  float d2 = 0.0;
  
  float max_lh = u_dense_height + 0.5 * u_dense_variation;
  if (uv.y > 1.0 - max_lh) {
  float height_noise = fbm(base_p * 1.5 + vec2(_time * 0.01, 0.0));
  float local_height = u_dense_height + (height_noise - 0.5) * u_dense_variation;
  float height_mask = smoothstep(1.0 - local_height, 1.0, uv.y);
  
  if (height_mask > 0.001) {
  vec2 p2 = base_p * u_scale_dense;
  p2.x += _time * 0.015;
  d2 = map_dense(p2, u_cov_dense) * height_mask;
  }
  }
  
  return max(d1, d2);
  }
  
*/

float get_clouds(vec2 uv, vec2 base_p) {
  float clump_mask = smoothstep(0.2, 0.7, fbm(base_p * 2.0 + _time * 0.002));
  
  if (clump_mask > 0.001) {
    vec2 p1 = base_p * u_scale_sparse * vec2(1.0, 3.0);
    p1.x += _time * 0.005;
    return map_sparse(p1, u_cov_sparse) * clump_mask * 0.7;
  }
  
  return 0.0;
}

void main() {
  vec2 uv = texture_coordinate;
  uv = floor(uv * u_pixel_size) / u_pixel_size;
  
  vec2 base_p = uv * (window_size / window_size.y);
  float d = get_clouds(uv, base_p);
  
  if (d <= 0.001) {
    o_attachment0 = vec4(u_col_cloud * custom_color.rgb, 0.0);
    return;
  }
  
  vec2 l_dir_offset = normalize(vec2(u_light_x, u_light_y)) * 0.05;
  float d_light = get_clouds(uv + l_dir_offset, base_p + l_dir_offset);
  
  float shadow = clamp((d_light - d) * u_shadow_str, 0.0, 1.0);
  float scatter = smoothstep(0.0, 0.2, d - d_light); 
  
  vec3 cloud = mix(u_col_cloud, u_col_shadow, shadow);
  cloud = mix(cloud, u_col_cloud, scatter); 
  
  float alpha = 1.0 - exp(-d * u_density);
  
  o_attachment0 = vec4(cloud * custom_color.rgb, alpha);
}