R"(
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
flat in uint flag;
in vec4 instance_color;

in vec2 offset;
in mat4 mv;

uniform sampler2D _t00;
uniform sampler2D _t02;
uniform vec3 lighting_ambient;

void main() {
  o_attachment0 = texture(_t00, texture_coordinate) * instance_color;
  //vec4 v = mv * vec4(gl_FragCoord.xy, 0, 1);
  //v /= v.w;
  //v/= 800;
  //v *= 2;
  //v -= 1;
  //vec2 frag_coord = v.xy;
  //frag_coord.y = 1.0 - frag_coord.y;

  vec4 t = vec4(texture(_t02, (gl_FragCoord.xy) / 800).rgb, 1);
  o_attachment0.rgb *= lighting_ambient + t.rgb;
}
)"