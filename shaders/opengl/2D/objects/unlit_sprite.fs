
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;

flat in uint fs_flags;

uniform sampler2D _t00;
uniform sampler2D _t01;

void main() {
  o_attachment0 = texture(_t00, texture_coordinate) * instance_color;
  if (o_attachment0.a <= 0.5) {
    discard;
  }

  if ((fs_flags & 0x2u) == 0x2u) {
   // o_attachment2 = o_attachment0;
  }
  else {
    //discard;
    //o_attachment2 = vec4(0, 0, 0, 0);
  }
}
