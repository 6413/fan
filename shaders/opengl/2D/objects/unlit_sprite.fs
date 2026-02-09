
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;

flat in uint fs_flags;

uniform sampler2D _t00;
uniform sampler2D _t01; // light buffer
uniform bool has_blending;

void main() {
  vec4 c = texture(_t00, texture_coordinate) * instance_color;
  if (!has_blending && c.a <= 0.5) {
    discard;
  }
  o_attachment0 = c;
}
