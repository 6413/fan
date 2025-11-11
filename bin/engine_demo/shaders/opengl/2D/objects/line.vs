#version 330 core
layout (location = 0) in vec4 color;
layout (location = 1) in vec3 src;
layout (location = 2) in vec2 dst;
layout (location = 3) in float line_thickness;

uniform mat4 view;
uniform mat4 projection;

out vec4 instance_color;
out vec2 line_start;
out vec2 line_end;
out float line_radius;
out vec3 frag_position;
out vec2 texture_coordinate;
out vec2 ba;
out float ba_len2;

vec2 rectangle_vertices[] = vec2[](
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
    vec2(-1.0, -1.0)
);

vec2 tc[] = vec2[](
    vec2(0, 0), // top left
    vec2(1, 0), // top right
    vec2(1, 1), // bottom right
    vec2(1, 1), // bottom right
    vec2(0, 1), // bottom left
    vec2(0, 0)  // top left
);

void main() {
  vec2 rp = rectangle_vertices[gl_VertexID % 6];
  texture_coordinate = tc[gl_VertexID % 6];
    
  vec2 start = src.xy;
  vec2 end = dst.xy;
  float radius = line_thickness * 0.5;
    
  vec2 capsule_dir = end - start;
  float capsule_length = length(capsule_dir);
  float angle = -atan(capsule_dir.y, capsule_dir.x);
    
  float box_margin = radius * 2.0;
  vec2 box_size = vec2(
      capsule_length + box_margin,
      box_margin                  
  );
    
  vec2 capsule_center = (start + end) * 0.5;
    
  mat2 rotation = mat2(
      cos(angle), -sin(angle),
      sin(angle), cos(angle)
  );
    
  vec2 local_pos = rp * (box_size * 0.5);
  vec2 rotated_pos = rotation * local_pos;
  vec2 world_pos = rotated_pos + capsule_center;
    
  frag_position = vec3(world_pos, src.z);
    
  instance_color = color;
  line_start = start;
  line_end = end;
  line_radius = radius;

  ba = line_end - line_start;
  ba_len2 = dot(ba, ba);

    
  gl_Position = projection * view * vec4(frag_position, 1.0);
}