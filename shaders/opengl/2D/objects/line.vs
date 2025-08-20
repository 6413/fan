#version 330 core

layout (location = 0) in vec4 color;
layout (location = 1) in vec3 src;
layout (location = 2) in vec2 dst;
layout (location = 3) in float line_thickness;

uniform mat4 view;
uniform mat4 projection;

out vec4 instance_color;
out vec2 line_coord;

void main() {
  uint vertex_id = uint(gl_VertexID);
    
  vec2 start = src.xy;
  vec2 end = dst.xy;
  vec2 direction = normalize(end - start);
  vec2 normal = vec2(-direction.y, direction.x);
  vec2 offset = normal * (line_thickness * 0.5);
    
  vec2 positions[6] = vec2[6](
      start + offset,   // top left
      start - offset,   // bottom left
      end + offset,     // top right
      start - offset,   // bottom left
      end - offset,     // bottom right
      end + offset      // top right
  );
  vec2 coords[6] = vec2[6](
    vec2(0.0, 1.0), 
    vec2(0.0, -1.0),
    vec2(1.0, 1.0), 
    vec2(0.0, -1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0)  
  );
  gl_Position = projection * view * vec4(positions[vertex_id], src.z, 1.0);
  instance_color = color;
  line_coord = coords[vertex_id];
}