
#version 130

in vec2 texture_coordinate;

in vec4 i_color;
in vec2 fragment_position;

out vec4 o_color;

uniform sampler2D texture_sampler;
uniform float iTime;

void main() {

  vec2 flipped = vec2(texture_coordinate.x, texture_coordinate.y);

  vec4 texture_color = texture(texture_sampler, flipped);

  float flicker = abs(sin(iTime)) * flipped.y;

  o_color = texture_color * i_color;
  o_color.r -= flicker;
  o_color.g += abs((flipped.y) * (texture_color * i_color).r);
}
