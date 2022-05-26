R"(
#version 330

in vec2 texture_coordinate;

in vec4 i_color;
in vec2 fragment_position;
in vec2 matrix_size;

out vec4 o_color;

uniform sampler2D texture_sampler;
uniform sampler2D texture_light_map;
uniform vec2 viewport_size;

void main() {

  vec2 flipped_y = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);

  vec4 texture_color = texture(texture_sampler, flipped_y);
  vec2 idk = gl_FragCoord.xy / viewport_size;
  vec4 light = texture(texture_light_map, vec2(idk.x, 1.0 - idk.y));

  o_color = texture_color * i_color;
  o_color.xyz *= light.x;
}
)"