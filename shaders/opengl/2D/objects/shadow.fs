#version 330

layout(location = 0) out vec4 o_attachment0;

const float CORE_OPACITY = 0.4;
const float FADE_END = 10000.0;
const float END_SOFTNESS = 1000.0;
const float ANGULAR_SOFTNESS = 0.1;

const float PI = 3.14159265359;
const float TWO_PI = 6.28318530718;
const float INV_PI = 0.31830988618;
const float EPS = 1e-6;
const float INF = 1e30;

flat in int shape;
in vec2 instance_position;
in vec2 instance_size;
in vec3 frag_position;
in vec2 light_position;
in float light_radius;
in vec4 instance_color;
in vec3 instance_angle;
in vec2 instance_rotation_point;

mat2 rotate2D(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

vec2 rotateAroundPoint(vec2 point, vec2 pivot, float angle) {
  mat2 rotation = rotate2D(angle);
  return pivot + rotation * (point - pivot);
}

float cross2(vec2 A, vec2 B) {
  return A.x * B.y - A.y * B.x;
}

bool rayOBB2D(vec2 O, vec2 D, vec2 center, vec2 half_size, float angle, out float t_enter, out float t_exit) {

  mat2 inv_rotation = rotate2D(-angle);
  vec2 local_O = inv_rotation * (O - center);
  vec2 local_D = inv_rotation * D;

  vec2 inv_D = 1.0 / (local_D + vec2(EPS * sign(local_D)));

  vec2 t1 = (-half_size - local_O) * inv_D;
  vec2 t2 = (half_size - local_O) * inv_D;

  vec2 tmin = min(t1, t2);
  vec2 tmax = max(t1, t2);

  t_enter = max(tmin.x, tmin.y);
  t_exit = min(tmax.x, tmax.y);

  return t_exit >= max(t_enter, 0.0);
}

bool pointInOBB(vec2 point, vec2 center, vec2 half_size, float angle) {

  mat2 inv_rotation = rotate2D(-angle);
  vec2 local_point = inv_rotation * (point - center);

  return abs(local_point.x) <= half_size.x && abs(local_point.y) <= half_size.y;
}

void getOBBCorners(vec2 center, vec2 half_size, float angle, out vec2 corners[4]) {
  mat2 rotation = rotate2D(angle);

  vec2 local_corners[4] = vec2[4](
    vec2(-half_size.x, -half_size.y),
    vec2(half_size.x, -half_size.y),
    vec2(half_size.x, half_size.y),
    vec2(-half_size.x, half_size.y)
  );

  for (int i = 0; i < 4; ++i) {
    corners[i] = center + rotation * local_corners[i];
  }
}

void silhouetteDirsOBB(vec2 L, vec2 center, vec2 half_size, float angle, out vec2 r0, out float d0, out vec2 r1, out float d1) {
  vec2 corners[4];
  getOBBCorners(center, half_size, angle, corners);

  vec2 dirs[4];
  float dists[4];
  for (int i = 0; i < 4; ++i) {
    vec2 diff = corners[i] - L;
    dists[i] = length(diff);
    dirs[i] = diff / dists[i];
  }

  vec2 mean = dirs[0] + dirs[1] + dirs[2] + dirs[3];
  float mlen = length(mean);
  mean = (mlen < EPS) ? vec2(1.0, 0.0) : mean / mlen;
  float baseAng = atan(mean.y, mean.x);

  float minOff =  1e9; int iMin = 0;
  float maxOff = -1e9; int iMax = 0;
  for (int i = 0; i < 4; ++i) {
    float ang = atan(dirs[i].y, dirs[i].x);
    float off = ang - baseAng;
    if (off <= -PI) off += TWO_PI;
    if (off >   PI) off -= TWO_PI;
    if (off < minOff) { minOff = off; iMin = i; }
    if (off > maxOff) { maxOff = off; iMax = i; }
  }

  r0 = dirs[iMin];
  d0 = dists[iMin];
  r1 = dirs[iMax];  
  d1 = dists[iMax];
}

float calculateAngularFade(vec2 V, vec2 r0, vec2 r1) {
  if (ANGULAR_SOFTNESS <= 0.0) {
    float cA = cross2(V, r0);
    float cB = cross2(V, r1);
    return float(cA <= 0.0 && cB >= 0.0);
  }

  vec2 v_norm = normalize(V);

  float angle_v = atan(v_norm.y, v_norm.x);
  float angle_r0 = atan(r0.y, r0.x);
  float angle_r1 = atan(r1.y, r1.x);

  angle_v = mod(angle_v + TWO_PI, TWO_PI);
  angle_r0 = mod(angle_r0 + TWO_PI, TWO_PI);
  angle_r1 = mod(angle_r1 + TWO_PI, TWO_PI);

  float left_angle = min(angle_r0, angle_r1);
  float right_angle = max(angle_r0, angle_r1);

  bool wraps = (right_angle - left_angle) > PI;
  if (wraps) {
    float temp = left_angle;
    left_angle = right_angle;
    right_angle = temp + TWO_PI;
    if (angle_v < right_angle - TWO_PI) {
      angle_v += TWO_PI;
    }
  }

  float fade = 1.0;
  if (angle_v >= left_angle && angle_v <= right_angle) {

    float dist_to_edge = min(angle_v - left_angle, right_angle - angle_v);
    fade = smoothstep(0.0, ANGULAR_SOFTNESS, dist_to_edge);
  } else {

    float dist_left = abs(angle_v - left_angle);
    float dist_right = abs(angle_v - right_angle);
    if (dist_left > PI) dist_left = TWO_PI - dist_left;
    if (dist_right > PI) dist_right = TWO_PI - dist_right;
    fade = 1.0 - smoothstep(0.0, ANGULAR_SOFTNESS, min(dist_left, dist_right));
  }

  return max(0.0, fade);
}

void main() {
  vec2 P = frag_position.xy;

  float rotation_angle = -instance_angle.z; 
  vec2 rotation_pivot = (length(instance_rotation_point) > EPS) ? instance_rotation_point : instance_position;

  if (pointInOBB(P, instance_position, instance_size, rotation_angle)) {
    o_attachment0 = vec4(0.0);
    return;
  }

  vec2 V = P - light_position;
  float dist_sq = dot(V, V);
  if (dist_sq < EPS * EPS) {
    o_attachment0 = vec4(0.0);
    return;
  }

  float dist = sqrt(dist_sq);
  vec2 dir = V / dist;

  float t_enter, t_exit;
  if (!rayOBB2D(light_position, dir, instance_position, instance_size, rotation_angle, t_enter, t_exit) || dist < t_enter) {
    o_attachment0 = vec4(0.0);
    return;
  }

  vec2 r0, r1;
  float d0, d1;
  silhouetteDirsOBB(light_position, instance_position, instance_size, rotation_angle, r0, d0, r1, d1);

  float angular_fade = calculateAngularFade(V, r0, r1);
  if (angular_fade <= 0.0) {
    o_attachment0 = vec4(0.0);
    return;
  }

  float t_max = max(d0, d1);
  float R_out = t_max + FADE_END;
  float R_in = R_out - END_SOFTNESS;

  float radial_fade = 1.0;
  if (dist >= R_out) {
    o_attachment0 = vec4(0.0);
    return;
  } else if (dist > R_in) {
    radial_fade = 1.0 - smoothstep(R_in, R_out, dist);
  }

  if (dist > t_max) {
    float wall_fade_dist = min(END_SOFTNESS, FADE_END * 0.1);
    float wall_fade = smoothstep(t_max, t_max + wall_fade_dist, dist);
    radial_fade = min(radial_fade, 1.0 - wall_fade);
  }

  float light_fall = clamp(1.0 - dist / light_radius, 0.0, 1.0);

  float alpha = CORE_OPACITY * radial_fade * light_fall * angular_fade;
  o_attachment0 = (alpha > 0.0) ? vec4(0.0, 0.0, 0.0, alpha) : vec4(0.0);
}