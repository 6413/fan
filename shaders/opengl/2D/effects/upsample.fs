#version 330

layout (location = 0) out vec3 o_color;

in vec2 texture_coordinate;

uniform float filter_radius;
uniform sampler2D _t00;

void main() {
  float x = filter_radius;
  float y = filter_radius;
    
  vec3 tl = texture(_t00, texture_coordinate + vec2(-x, y)).rgb;
  vec3 tr = texture(_t00, texture_coordinate + vec2(x, y)).rgb;
  vec3 bl = texture(_t00, texture_coordinate + vec2(-x, -y)).rgb;
  vec3 br = texture(_t00, texture_coordinate + vec2(x, -y)).rgb;
    
  o_color = (tl + tr + bl + br) * 0.25;
}