#version 450
vec2 verts[6] = vec2[](vec2(1,-1), vec2(1,1), vec2(-1,-1), vec2(-1,-1), vec2(1,1), vec2(-1,1));
layout(location = 0) out vec2 v_uv;
void main() {
  vec2 p = verts[gl_VertexIndex];
  v_uv = p * 0.5 + 0.5;
  gl_Position = vec4(p, 0.0, 1.0);
}
