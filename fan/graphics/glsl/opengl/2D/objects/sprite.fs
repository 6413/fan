
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform vec3 lighting_ambient;
uniform vec2 window_size;



void main() {
  o_attachment0 = texture(_t00, texture_coordinate) * instance_color;

  if (o_attachment0.a <= 0.5) {
    discard;
  }

  vec4 t = vec4(texture(_t01, gl_FragCoord.xy / window_size).rgb, 1);

  o_attachment0.rgb *= lighting_ambient + t.rgb;
}
