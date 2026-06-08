#version 430 core

uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 out_color;

void main() { 
  out_color = texture(u_tex, v_uv); 
}