R"(
#version 330

in vec2 texture_coordinate;

in vec4 i_color;
in vec2 fragment_position;

out vec4 o_color;

uniform sampler2D texture_sampler;

void main() {
  o_color = texture(texture_sampler, texture_coordinate) * i_color;
}
)"