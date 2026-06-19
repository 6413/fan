#version 330 core
in vec2 v_uv;
uniform sampler2D sprite_texture;
uniform float alpha_threshold;
out vec4 out_color;

void main() {
  float a = texture(sprite_texture, v_uv).a;
  float b = float(a > alpha_threshold);
  out_color = vec4(b, b, b, 1.0);
}