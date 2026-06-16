#version 420

layout(location = 0) out vec2 texture_coordinate;

vec2 rectangle_vertices[] = vec2[](
  vec2(1.0, -1.0),
  vec2(1.0, 1.0),
  vec2(-1.0, -1.0),
  vec2(-1.0, -1.0),
  vec2(1.0, 1.0),
  vec2(-1.0, 1.0)
);

void main() {
  vec2 p = rectangle_vertices[gl_VertexIndex];
  texture_coordinate = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}
