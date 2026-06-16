#version 420

layout(set = 0, binding = 0) uniform sampler2D _t00;

layout(push_constant) uniform push_constants_t {
  vec4 filter_radius;
} pc;

layout(location = 0) in vec2 texture_coordinate;
layout(location = 0) out vec4 o_color;

void main() {
  float x = pc.filter_radius.x;
  float y = pc.filter_radius.y;

  vec3 a = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y + y)).rgb;
  vec3 b = texture(_t00, vec2(texture_coordinate.x,     texture_coordinate.y + y)).rgb;
  vec3 c = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y + y)).rgb;

  vec3 d = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y)).rgb;
  vec3 e = texture(_t00, vec2(texture_coordinate.x,     texture_coordinate.y)).rgb;
  vec3 f = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y)).rgb;

  vec3 g = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y - y)).rgb;
  vec3 h = texture(_t00, vec2(texture_coordinate.x,     texture_coordinate.y - y)).rgb;
  vec3 i = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y - y)).rgb;

  vec3 color = e * 4.0;
  color += (b + d + f + h) * 2.0;
  color += (a + c + g + i);
  color *= 1.0 / 16.0;
  o_color = vec4(color, 1.0);
}
