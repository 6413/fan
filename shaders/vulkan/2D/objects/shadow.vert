#version 450

layout(std430, set = 0, binding = 0) readonly buffer instances_t {
  uint raw_data[];
};

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
} constants;

struct pv_t {
  mat4 projection;
  mat4 view;
};

layout(set = 0, binding = 1) uniform upv_t {
  pv_t pv[16];
};

uint base;
float f(uint o) { return uintBitsToFloat(raw_data[base + o / 4u]); }
uint u(uint o) { return raw_data[base + o / 4u]; }
int i(uint o) { return int(raw_data[base + o / 4u]); }
vec2 v2(uint o) { return vec2(f(o), f(o + 4u)); }
vec3 v3(uint o) { return vec3(f(o), f(o + 4u), f(o + 8u)); }
vec4 v4(uint o) { return vec4(f(o), f(o + 4u), f(o + 8u), f(o + 12u)); }

vec2 rectangle_vertices[6] = vec2[](
  vec2(-1.0, -1.0),
  vec2(1.0, -1.0),
  vec2(1.0, 1.0),
  vec2(1.0, 1.0),
  vec2(-1.0, 1.0),
  vec2(-1.0, -1.0)
);

vec2 texture_coordinates[6] = vec2[](
  vec2(0.0, 0.0),
  vec2(1.0, 0.0),
  vec2(1.0, 1.0),
  vec2(1.0, 1.0),
  vec2(0.0, 1.0),
  vec2(0.0, 0.0)
);

vec2 rotate2(vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

vec2 rotate_around(vec2 p, vec2 pivot, float a) {
  return pivot + rotate2(p - pivot, a);
}

vec4 project2(vec3 p) {
  return pv[constants.camera_id].projection * pv[constants.camera_id].view * vec4(p, 1.0);
}

layout(location = 0) out vec4 instance_color;
layout(location = 1) out vec2 instance_position;
layout(location = 2) out vec2 instance_size;
layout(location = 3) out vec3 frag_position;
layout(location = 4) flat out int shape;
layout(location = 5) out vec2 light_position;
layout(location = 6) out float light_radius;
layout(location = 7) out vec3 instance_angle;
layout(location = 8) out vec2 instance_rotation_point;

const float fade_end = 10000.0;
const float pi = 3.14159265359;

void main() {
  base = uint(gl_InstanceIndex) * 20u;
  uint id = uint(gl_VertexIndex) % 6u;
  vec2 rp = rectangle_vertices[id];
  vec3 position = v3(0u);
  vec2 size = v2(16u);
  vec2 rotation_point = v2(24u);
  vec3 angle = v3(52u);
  vec2 light = v2(64u);
  vec2 wall_min = position.xy - size;
  vec2 wall_max = position.xy + size;
  vec2 corners[4] = vec2[4](wall_min, vec2(wall_max.x, wall_min.y), wall_max, vec2(wall_min.x, wall_max.y));
  vec2 pivot = length(rotation_point) > 0.000001 ? rotation_point : position.xy;
  for (int j = 0; j < 4; ++j) {
    corners[j] = rotate_around(corners[j], pivot, angle.z);
  }
  vec2 min_corner = corners[0];
  vec2 max_corner = corners[0];
  for (int j = 1; j < 4; ++j) {
    min_corner = min(min_corner, corners[j]);
    max_corner = max(max_corner, corners[j]);
  }
  vec2 dirs[4];
  for (int j = 0; j < 4; ++j) {
    dirs[j] = normalize(corners[j] - light);
  }
  vec2 mean = dirs[0] + dirs[1] + dirs[2] + dirs[3];
  float mlen = length(mean);
  mean = mlen < 0.000001 ? vec2(1.0, 0.0) : mean / mlen;
  float base_angle = atan(mean.y, mean.x);
  float min_off = 1000000000.0;
  float max_off = -1000000000.0;
  int min_i = 0;
  int max_i = 0;
  for (int j = 0; j < 4; ++j) {
    float a = atan(dirs[j].y, dirs[j].x);
    float off = a - base_angle;
    if (off <= -pi) { off += 2.0 * pi; }
    if (off > pi) { off -= 2.0 * pi; }
    if (off < min_off) { min_off = off; min_i = j; }
    if (off > max_off) { max_off = off; max_i = j; }
  }
  float max_dist = max(distance(corners[min_i], light), distance(corners[max_i], light));
  vec2 shadow_end0 = light + dirs[min_i] * (max_dist + fade_end);
  vec2 shadow_end1 = light + dirs[max_i] * (max_dist + fade_end);
  vec2 bbox_min = min(min(min_corner, shadow_end0), min(shadow_end1, light));
  vec2 bbox_max = max(max(max_corner, shadow_end0), max(shadow_end1, light));
  float padding = f(72u) * 0.1;
  bbox_min -= padding;
  bbox_max += padding;
  vec2 center = (bbox_min + bbox_max) * 0.5;
  vec2 half_size = (bbox_max - bbox_min) * 0.5;
  vec2 world = center + rp * half_size;
  instance_color = v4(32u);
  instance_position = position.xy;
  instance_size = size;
  frag_position = vec3(world, position.z);
  shape = i(12u);
  light_position = light;
  light_radius = f(72u);
  instance_angle = angle;
  instance_rotation_point = rotation_point;
  gl_Position = project2(frag_position);
}