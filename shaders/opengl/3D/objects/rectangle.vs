#version 120

#extension GL_EXT_gpu_shader4 : enable

varying vec4 instance_color;
uniform mat4 view;
uniform mat4 projection;
attribute vec3 in_position;
attribute vec3 in_size;
attribute vec4 in_color;

vec3 get_rectangle_vertex(int index) {
  // Front face
  if (index == 0) return vec3(-1.0, -1.0, 1.0);
  if (index == 1) return vec3(1.0, -1.0, 1.0);
  if (index == 2) return vec3(1.0, 1.0, 1.0);
  if (index == 3) return vec3(1.0, 1.0, 1.0);
  if (index == 4) return vec3(-1.0, 1.0, 1.0);
  if (index == 5) return vec3(-1.0, -1.0, 1.0);
    
  // Back face
  if (index == 6) return vec3(-1.0, -1.0, -1.0);
  if (index == 7) return vec3(1.0, -1.0, -1.0);
  if (index == 8) return vec3(1.0, 1.0, -1.0);
  if (index == 9) return vec3(1.0, 1.0, -1.0);
  if (index == 10) return vec3(-1.0, 1.0, -1.0);
  if (index == 11) return vec3(-1.0, -1.0, -1.0);
    
  // Left face
  if (index == 12) return vec3(-1.0, -1.0, -1.0);
  if (index == 13) return vec3(-1.0, -1.0, 1.0);
  if (index == 14) return vec3(-1.0, 1.0, 1.0);
  if (index == 15) return vec3(-1.0, 1.0, 1.0);
  if (index == 16) return vec3(-1.0, 1.0, -1.0);
  if (index == 17) return vec3(-1.0, -1.0, -1.0);
    
  // Right face
  if (index == 18) return vec3(1.0, -1.0, -1.0);
  if (index == 19) return vec3(1.0, -1.0, 1.0);
  if (index == 20) return vec3(1.0, 1.0, 1.0);
  if (index == 21) return vec3(1.0, 1.0, 1.0);
  if (index == 22) return vec3(1.0, 1.0, -1.0);
  if (index == 23) return vec3(1.0, -1.0, -1.0);
    
  // Top face
  if (index == 24) return vec3(-1.0, 1.0, 1.0);
  if (index == 25) return vec3(1.0, 1.0, 1.0);
  if (index == 26) return vec3(1.0, 1.0, -1.0);
  if (index == 27) return vec3(1.0, 1.0, -1.0);
  if (index == 28) return vec3(-1.0, 1.0, -1.0);
  if (index == 29) return vec3(-1.0, 1.0, 1.0);
    
  // Bottom face
  if (index == 30) return vec3(-1.0, -1.0, 1.0);
  if (index == 31) return vec3(1.0, -1.0, 1.0);
  if (index == 32) return vec3(1.0, -1.0, -1.0);
  if (index == 33) return vec3(1.0, -1.0, -1.0);
  if (index == 34) return vec3(-1.0, -1.0, -1.0);
  if (index == 35) return vec3(-1.0, -1.0, 1.0);
    
  // Default case (shouldn't happen)
  return vec3(0.0, 0.0, 0.0);
}

void main() {
    int id = gl_VertexID - (gl_VertexID / 36) * 36; // Modulo operation for GLSL 120
    vec3 rp = get_rectangle_vertex(id);
    mat4 m = mat4(1.0);
    vec3 rotated = vec3(m * vec4(rp * in_size + in_position, 1.0));
    gl_Position = projection * view * vec4(rotated, 1.0);
    instance_color = in_color;
}