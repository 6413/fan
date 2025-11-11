#version 330
layout (location = 0) in vec3 in_position;
layout (location = 1) in int in_shape;
layout (location = 2) in vec2 in_size;
layout (location = 3) in vec2 in_rotation_point;
layout (location = 4) in vec4 in_color;
layout (location = 5) in uint in_flags;
layout (location = 6) in vec3 in_angle;  
layout (location = 7) in vec2 in_light_position;
layout (location = 8) in float in_light_radius;

out vec4 instance_color;
out vec2 instance_position;
out vec2 instance_size;
out vec3 frag_position;
out vec2 uv;
flat out int shape;
out vec2 light_position;
out float light_radius;
out vec2 texture_coordinate;
flat out uint fs_flags;
out vec3 instance_angle;
out vec2 instance_rotation_point;

uniform mat4 view;
uniform mat4 projection;

vec2 rectangle_vertices[] = vec2[](
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
    vec2(-1.0, -1.0)
);

const float FADE_END = 10000.0;
const float PI = 3.14159265359;

mat2 rotate2D(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat2(c, -s, s, c);
}

vec2 rotateAroundPoint(vec2 point, vec2 pivot, float angle) {
  mat2 rotation = rotate2D(angle);
  return pivot + rotation * (point - pivot);
}

void main() {
  uint id = uint(gl_VertexID % 6);
  vec2 rp = rectangle_vertices[id];

  float rotation_angle = in_angle.z;

  vec2 rotation_pivot = (length(in_rotation_point) > 1e-6) ? in_rotation_point : in_position.xy;

  vec2 wall_min = in_position.xy - in_size;
  vec2 wall_max = in_position.xy + in_size;

  vec2 local_corners[4] = vec2[4](
      wall_min,
      vec2(wall_max.x, wall_min.y),
      wall_max,
      vec2(wall_min.x, wall_max.y)
  );

  vec2 rotated_corners[4];
  for (int i = 0; i < 4; ++i) {
      rotated_corners[i] = rotateAroundPoint(local_corners[i], rotation_pivot, rotation_angle);
  }

  vec2 rotated_wall_min = rotated_corners[0];
  vec2 rotated_wall_max = rotated_corners[0];
  for (int i = 1; i < 4; ++i) {
      rotated_wall_min = min(rotated_wall_min, rotated_corners[i]);
      rotated_wall_max = max(rotated_wall_max, rotated_corners[i]);
  }

  vec2 L = in_light_position;
  vec2 dirs[4];
  for (int i = 0; i < 4; ++i) {
      dirs[i] = normalize(rotated_corners[i] - L);
  }

  vec2 mean = dirs[0] + dirs[1] + dirs[2] + dirs[3];
  float mlen = length(mean);
  mean = (mlen < 1e-6) ? vec2(1.0, 0.0) : mean / mlen;
  float baseAng = atan(mean.y, mean.x);

  float minOff = 1e9; int iMin = 0;
  float maxOff = -1e9; int iMax = 0;
  for (int i = 0; i < 4; ++i) {
      float ang = atan(dirs[i].y, dirs[i].x);
      float off = ang - baseAng;
      if (off <= -PI) off += 2.0 * PI;
      if (off > PI) off -= 2.0 * PI;
      if (off < minOff) { minOff = off; iMin = i; }
      if (off > maxOff) { maxOff = off; iMax = i; }
  }

  vec2 silhouette_dir1 = dirs[iMin];
  vec2 silhouette_dir2 = dirs[iMax];
  float max_corner_dist = max(distance(rotated_corners[iMin], L), distance(rotated_corners[iMax], L));

  float shadow_end = max_corner_dist + FADE_END;

  vec2 shadow_end1 = L + silhouette_dir1 * shadow_end;
  vec2 shadow_end2 = L + silhouette_dir2 * shadow_end;

  vec2 bbox_min = min(min(rotated_wall_min, min(shadow_end1, shadow_end2)), L);
  vec2 bbox_max = max(max(rotated_wall_max, max(shadow_end1, shadow_end2)), L);

  float padding = in_light_radius * 0.1;
  bbox_min -= padding;
  bbox_max += padding;

  vec2 bbox_center = (bbox_min + bbox_max) * 0.5;
  vec2 bbox_size = (bbox_max - bbox_min) * 0.5;

  vec2 world_pos = bbox_center + rp * bbox_size;

  instance_position = in_position.xy;  
  instance_size = in_size;              
  frag_position = vec3(world_pos, in_position.z);
  uv = rp;
  gl_Position = projection * view * vec4(world_pos, in_position.z, 1.0);
  instance_color = in_color;
  fs_flags = in_flags;
  light_position = in_light_position;
  light_radius = in_light_radius;
  instance_angle = in_angle;
  instance_rotation_point = in_rotation_point;
  shape = in_shape;
}