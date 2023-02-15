R"(
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
flat in uint flag;
in vec4 instance_color;

in mat4 mv;

uniform sampler2D _t00;
uniform sampler2D _t02;
uniform vec3 lighting_ambient;
uniform vec2 window_size;

void main() {
  o_attachment0 = texture(_t00, texture_coordinate) * instance_color;
  //vec4 t = vec4(texture(_t02, gl_FragCoord.xy / window_size).rgb, 1);
  //o_attachment0.rgb *= lighting_ambient + t.rgb;
}
)"