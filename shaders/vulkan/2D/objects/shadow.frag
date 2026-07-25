#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 instance_position;
layout(location = 2) in vec2 instance_size;
layout(location = 3) in vec3 frag_position;
layout(location = 4) flat in int shape;
layout(location = 5) in vec2 light_position;
layout(location = 6) in float light_radius;
layout(location = 7) in vec3 instance_angle;
layout(location = 8) in vec2 instance_rotation_point;
layout(location = 0) out vec4 o_attachment0;

const float core_opacity = 0.4;
const float fade_end = 10000.0;
const float end_softness = 1000.0;
const float angular_softness = 0.1;
const float pi = 3.14159265359;
const float two_pi = 6.28318530718;
const float eps = 0.000001;

vec2 rotate2(vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

float cross2(vec2 a, vec2 b) {
  return a.x * b.y - a.y * b.x;
}

void ray_circle(vec2 o, vec2 d, vec2 center, float radius, out float t_enter, out float t_exit) {
  vec2 oc = o - center;
  float b = dot(oc, d);
  float c = dot(oc, oc) - radius * radius;
  float h = b * b - c;
  if (h < 0.0) {
    t_enter = max(-b, 0.0);
    t_exit = t_enter;
    return;
  }
  h = sqrt(h);
  t_enter = -b - h;
  t_exit = -b + h;
}

void ray_obb(vec2 o, vec2 d, vec2 center, vec2 half_size, float angle, out float t_enter, out float t_exit) {
  vec2 local_o = rotate2(o - center, -angle);
  vec2 local_d = rotate2(d, -angle);
  vec2 eps_dir = vec2(local_d.x >= 0.0 ? eps : -eps, local_d.y >= 0.0 ? eps : -eps);
  vec2 inv_d = 1.0 / (local_d + eps_dir);
  vec2 t0 = (-half_size - local_o) * inv_d;
  vec2 t1 = (half_size - local_o) * inv_d;
  vec2 tmin = min(t0, t1);
  vec2 tmax = max(t0, t1);
  t_enter = max(tmin.x, tmin.y);
  t_exit = min(tmax.x, tmax.y);
  if (t_exit < max(t_enter, 0.0)) {
    t_enter = max(dot(center - o, d), 0.0);
    t_exit = t_enter;
  }
}

bool point_in_circle(vec2 p, vec2 center, float radius) {
  vec2 diff = p - center;
  return dot(diff, diff) <= radius * radius;
}

bool point_in_obb(vec2 p, vec2 center, vec2 half_size, float angle) {
  vec2 local_p = rotate2(p - center, -angle);
  return abs(local_p.x) <= half_size.x && abs(local_p.y) <= half_size.y;
}

void get_corners(vec2 center, vec2 half_size, float angle, out vec2 corners[4]) {
  vec2 local[4] = vec2[4](vec2(-half_size.x, -half_size.y), vec2(half_size.x, -half_size.y), half_size, vec2(-half_size.x, half_size.y));
  for (int j = 0; j < 4; ++j) {
    corners[j] = center + rotate2(local[j], angle);
  }
}

void silhouette_dirs_circle(vec2 light, vec2 center, float radius, out vec2 r0, out float d0, out vec2 r1, out float d1) {
  vec2 diff = center - light;
  float dist = length(diff);
  vec2 dir = diff / max(dist, eps);
  float theta = asin(clamp(radius / dist, 0.0, 1.0));
  r0 = rotate2(dir, -theta);
  r1 = rotate2(dir, theta);
  d0 = dist + radius;
  d1 = d0;
}

void silhouette_dirs(vec2 light, vec2 center, vec2 half_size, float angle, out vec2 r0, out float d0, out vec2 r1, out float d1) {
  vec2 corners[4];
  get_corners(center, half_size, angle, corners);
  vec2 dirs[4];
  float dists[4];
  for (int j = 0; j < 4; ++j) {
    vec2 diff = corners[j] - light;
    dists[j] = length(diff);
    dirs[j] = diff / max(dists[j], eps);
  }
  vec2 to_center = center - light;
  float base_angle = atan(to_center.y, to_center.x);
  float min_off = 1000000000.0;
  float max_off = -1000000000.0;
  int min_i = 0;
  int max_i = 0;
  for (int j = 0; j < 4; ++j) {
    float a = atan(dirs[j].y, dirs[j].x);
    float off = a - base_angle;
    if (off <= -pi) { off += two_pi; }
    if (off > pi) { off -= two_pi; }
    if (off < min_off) { min_off = off; min_i = j; }
    if (off > max_off) { max_off = off; max_i = j; }
  }
  r0 = dirs[min_i];
  d0 = dists[min_i];
  r1 = dirs[max_i];
  d1 = dists[max_i];
}

float angular_fade(vec2 v, vec2 r0, vec2 r1) {
  const float cone_margin = 0.0015;
  vec2 vn = normalize(v);
  float av = mod(atan(vn.y, vn.x) + two_pi, two_pi);
  float a0 = mod(atan(r0.y, r0.x) + two_pi, two_pi);
  float a1 = mod(atan(r1.y, r1.x) + two_pi, two_pi);
  float l = min(a0, a1) - cone_margin;
  float r = max(a0, a1) + cone_margin;
  if (r - l > pi) {
    float t = l;
    l = r;
    r = t + two_pi;
    if (av < r - two_pi) { av += two_pi; }
  }
  if (av >= l && av <= r) {
    return 1.0;
  }
  float dist_to_cone = min(abs(av - l), abs(av - r));
  return 1.0 - smoothstep(0.0, angular_softness, dist_to_cone);
}

void main() {
  vec2 p = frag_position.xy;
  float angle = -instance_angle.z;
  float radius = instance_size.x;
  bool is_circle = shape == 1;

  bool inside = is_circle ? point_in_circle(p, instance_position, radius) : point_in_obb(p, instance_position, instance_size, angle);
  if (inside) {
    o_attachment0 = vec4(0.0);
    return;
  }
  
  vec2 v = p - light_position;
  float dist_sq = dot(v, v);
  if (dist_sq < eps * eps) {
    o_attachment0 = vec4(0.0);
    return;
  }
  
  float dist = sqrt(dist_sq);
  vec2 dir = v / dist;
  float t_enter;
  float t_exit;
  
  if (is_circle) {
    ray_circle(light_position, dir, instance_position, radius, t_enter, t_exit);
  } else {
    ray_obb(light_position, dir, instance_position, instance_size, angle, t_enter, t_exit);
  }

  if (dist < t_enter) {
    o_attachment0 = vec4(0.0);
    return;
  }
  
  vec2 r0;
  vec2 r1;
  float d0;
  float d1;
  if (is_circle) {
    silhouette_dirs_circle(light_position, instance_position, radius, r0, d0, r1, d1);
  } else {
    silhouette_dirs(light_position, instance_position, instance_size, angle, r0, d0, r1, d1);
  }
  
  float af = angular_fade(v, r0, r1);
  if (af <= 0.0) {
    o_attachment0 = vec4(0.0);
    return;
  }
  float t_max = max(d0, d1);
  float r_out = t_max + fade_end;
  float r_in = r_out - end_softness;
  if (dist >= r_out) {
    o_attachment0 = vec4(0.0);
    return;
  }
  float radial = dist > r_in ? 1.0 - smoothstep(r_in, r_out, dist) : 1.0;
  if (dist > t_max) {
    radial = min(radial, 1.0 - smoothstep(t_max, t_max + min(end_softness, fade_end * 0.1), dist));
  }
  float light_fall = clamp(1.0 - dist / max(light_radius, eps), 0.0, 1.0);
  float alpha = core_opacity * radial * light_fall * af;
  o_attachment0 = alpha > 0.0 ? vec4(instance_color.rgb, alpha * instance_color.a) : vec4(0.0);
}