
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec2 size;
in vec4 instance_color;
flat in int fs_flags;
flat in int element_id;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform vec3 lighting_ambient;
uniform vec2 window_size;
uniform float m_time;

void main() {

  vec2 tc = texture_coordinate;

  if (fs_flags == 1) {
    vec2 Wave = vec2(10, 2);
    tc += vec2(cos((tc.y/Wave.x+(m_time + (m_time * (float(element_id))) / 100.f))*2.2831)*Wave.y,0)/size*(1.0-tc.y);
  }

  o_attachment0 = texture(_t00, tc) * instance_color;

  if (o_attachment0.a <= 0.25) {
    discard;
  }

  vec4 t = vec4(texture(_t01, gl_FragCoord.xy / window_size).rgb, 1);

  o_attachment0.rgb *= lighting_ambient + t.rgb;
  //o_attachment0.rgb *= lighting_ambient;
  //o_attachment0.rgb += t.rgb;
}
