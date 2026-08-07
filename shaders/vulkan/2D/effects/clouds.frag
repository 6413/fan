#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 2) in flat vec2 object_size;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
  uint texture_id1;
  uint texture_id2;
  uint texture_id3;
  float time;
  uint lightmap_id;
  float ambient_floor;
  vec4 lighting_ambient;
} constants;

layout(set = 0, binding = 3) uniform u_t {
  float u_pixel_size; // 1200
  float u_scale_sparse; // 4.5
  float u_cov_sparse; // 0.5
  float u_warp_sparse; // 2.1
  float u_scale_dense; // 3.8
  float u_cov_dense; // 0.4
  float u_warp_dense; // 3.7
  float u_dense_height; // 0.5
  float u_dense_puffiness; // 2.0
  float u_dense_variation; // 0.4
  float u_density; // 4.7
  float u_shadow_str; // 2.5
  float u_light_x; // 0
  float u_light_y; // 1
  float u_scroll_x; // 0.01 min -1 max 1 step 0.01
  float u_edge_smooth; // 0.15
  float u_scale_mass; // 1.3
  float u_cov_mass; // 0.3
  float u_warp_mass; // 1.6
  float u_mass_amount; // 0.4 min 0 max 1 step 0.05
  float u_scale_detail; // 6.5
  float u_cov_detail; // 0.5
  float u_warp_detail; // 2.6
  float u_detail_amount; // 0.18 min 0 max 1 step 0.05
  float u_variation_scale; // 2.2
  float u_variation_strength; // 0.6 min 0 max 2 step 0.05
  vec3 u_col_sky_top; // color 0.10 0.29 0.47
  vec3 u_col_sky_bot; // color 0.42 0.69 0.90
  vec3 u_col_cloud; // color 1.0 1.0 1.0
  vec3 u_col_shadow; // color 0.48 0.58 0.66
} u;

const mat2 fbm_mat = mat2(1.6, 1.2, -1.2, 1.6);
const vec2 hash_c1 = vec2(127.1, 311.7);
const vec2 hash_c2 = vec2(269.5, 183.3);

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 uu = f * f * (3.0 - 2.0 * f);

  vec2 i00 = i;
  vec2 i10 = i + vec2(1.0, 0.0);
  vec2 i01 = i + vec2(0.0, 1.0);
  vec2 i11 = i + vec2(1.0, 1.0);

  vec2 h00 = fract(sin(vec2(dot(i00, hash_c1), dot(i00, hash_c2))) * 43758.5453123) * 2.0 - 1.0;
  vec2 h10 = fract(sin(vec2(dot(i10, hash_c1), dot(i10, hash_c2))) * 43758.5453123) * 2.0 - 1.0;
  vec2 h01 = fract(sin(vec2(dot(i01, hash_c1), dot(i01, hash_c2))) * 43758.5453123) * 2.0 - 1.0;
  vec2 h11 = fract(sin(vec2(dot(i11, hash_c1), dot(i11, hash_c2))) * 43758.5453123) * 2.0 - 1.0;

  return mix(
    mix(dot(h00, f), dot(h10, f - vec2(1.0, 0.0)), uu.x),
    mix(dot(h01, f - vec2(0.0, 1.0)), dot(h11, f - vec2(1.0, 1.0)), uu.x),
    uu.y
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

float fbm3(vec2 p) {
  float f = 0.0, amp = 0.5;
  for (int i = 0; i < 3; i++) {
    f += amp * noise(p);
    p = fbm_mat * p;
    amp *= 0.5;
  }
  return f;
}

float map_cloud(vec2 p, float coverage, float soft, float warp) {
  vec2 q = vec2(fbm(p), fbm(p + vec2(5.2, 1.3)));
  return smoothstep(1.0 - coverage - soft, 1.0, fbm(p + q * warp));
}

float map_cloud3(vec2 p, float coverage, float soft, float warp) {
  vec2 q = vec2(fbm3(p), fbm3(p + vec2(5.2, 1.3)));
  return smoothstep(1.0 - coverage - soft, 1.0, fbm3(p + q * warp));
}

float get_clouds(vec2 uv, vec2 base_p) {
  float ambient = clamp(constants.ambient_floor + dot(constants.lighting_ambient.rgb, vec3(0.299, 0.587, 0.114)), 0.0, 1.0);
  float soft = u.u_edge_smooth * (0.5 + 0.5 * ambient);
  float h = clamp(u.u_dense_height, 0.0, 1.0);
  float hv = max(u.u_dense_variation, 0.001);
  float height_mask = 1.0 - smoothstep(h, h + hv, base_p.y);

  // regional variation — slow independent fields so some areas stay clear
  // while others suddenly pack full of big banks
  float vs = u.u_variation_strength;
  float v = fbm3(base_p * u.u_variation_scale + vec2(11.3, 5.7));
  float v2 = fbm3(base_p * u.u_variation_scale * 1.7 + vec2(3.1, 9.2));

  float cov_sparse = clamp(u.u_cov_sparse + (v - 0.5) * vs * 1.2, 0.02, 0.98);
  float cov_dense = clamp(u.u_cov_dense + (v2 - 0.5) * vs * 0.8, 0.02, 0.98);
  float cov_mass = clamp(u.u_cov_mass + (v - 0.5) * vs * 2.4, 0.02, 0.98);
  float cov_detail = clamp(u.u_cov_detail + (v2 - 0.5) * vs * 1.6, 0.02, 0.98);

  float clump_mask = smoothstep(0.2 - soft, 0.7, fbm(base_p * 2.0 + constants.time * 0.002));
  // independent low-frequency gate so the big banks can span whole areas
  float mass_mask = smoothstep(0.15 - soft, 0.65, fbm3(base_p * 0.9 + constants.time * 0.001 + vec2(7.3, 2.9)));

  vec2 p1 = base_p * u.u_scale_sparse * vec2(1.0, 3.0);
  p1.x += constants.time * 0.005;
  float d = map_cloud(p1, cov_sparse, soft, u.u_warp_sparse) * clump_mask * 0.5;

  vec2 p2 = base_p * u.u_scale_dense * vec2(1.0, 2.0);
  p2.x += constants.time * 0.01;
  float dense = map_cloud(p2, cov_dense, soft, u.u_warp_dense);
  d += pow(dense, u.u_dense_puffiness) * clump_mask * 0.12;

  // huge slow banks that appear regionally
  vec2 p3 = base_p * u.u_scale_mass * vec2(1.0, 2.5);
  p3.x += constants.time * 0.002;
  float mass = map_cloud3(p3, cov_mass, soft * 1.6, u.u_warp_mass);
  d += mass * mass_mask * u.u_mass_amount;

  // small scattered puffs, also present in mostly clear areas
  vec2 p4 = base_p * u.u_scale_detail * vec2(1.0, 1.6);
  p4.x += constants.time * 0.02;
  float detail = map_cloud3(p4, cov_detail, soft, u.u_warp_detail);
  d += detail * (0.4 + 0.6 * clump_mask) * u.u_detail_amount;

  return d * height_mask;
}

void main() {
  vec2 uv = texture_coordinate;
  uv = floor(uv * u.u_pixel_size) / u.u_pixel_size;

  vec2 aspect = object_size / object_size.y;
  vec2 base_p = uv * aspect;
  base_p.x += u.u_scroll_x * constants.time;
  float d = get_clouds(uv, base_p);

  if (d <= 0.001) {
    o_attachment0 = vec4(u.u_col_cloud * instance_color.rgb, 0.0);
    return;
  }

  vec2 l_dir = vec2(u.u_light_x, u.u_light_y);
  if (dot(l_dir, l_dir) < 0.0001) { l_dir = vec2(0.0, 1.0); }
  vec2 l_dir_offset = normalize(l_dir) * 0.05;
  float d_light = get_clouds(uv + l_dir_offset, base_p + l_dir_offset);

  float shadow = clamp((d_light - d) * u.u_shadow_str, 0.0, 1.0);
  float scatter = smoothstep(0.0, 0.2, d - d_light);

  float ambient = clamp(constants.ambient_floor + dot(constants.lighting_ambient.rgb, vec3(0.299, 0.587, 0.114)), 0.0, 1.0);
  vec3 sky_col = mix(u.u_col_sky_bot, u.u_col_sky_top, clamp(base_p.y, 0.0, 1.0));

  vec3 cloud = mix(u.u_col_cloud, u.u_col_shadow, shadow);
  cloud = mix(cloud, u.u_col_cloud, scatter);
  cloud = mix(cloud, sky_col, 0.2 * ambient);

  float edge_ramp = smoothstep(0.0, 0.15 + u.u_edge_smooth * 0.85, d);
  float alpha = (1.0 - exp(-d * u.u_density * 0.5)) * edge_ramp;

  o_attachment0 = vec4(cloud * instance_color.rgb, alpha);
}
