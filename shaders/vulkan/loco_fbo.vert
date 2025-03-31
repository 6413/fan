#version 420

vec2 rectangle_vertices[] = vec2[](
  vec2(1.0, -1.0),
  
  vec2(1.0, 1.0),
  vec2(-1.0, -1.0),
  
  vec2(-1.0, -1.0),
  vec2(1.0, 1.0),
  vec2(-1.0, 1.0)
  
);

void main() {

  gl_Position = vec4(rectangle_vertices[gl_VertexIndex], 0, 1);
}