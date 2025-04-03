
#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec2 size;
in vec4 instance_color;
flat in uint fs_flags;
flat in float object_seed;

in vec3 instance_position;
in vec3 frag_position;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;
uniform vec3 lighting_ambient;
uniform vec2 window_size;
uniform float _time;
uniform vec2 offset;


void main() {

  vec2 tc = texture_coordinate;

  vec4 tex_color = vec4(1, 1, 1, 1);

  tex_color = texture(_t00, tc) * instance_color;

  if (tex_color.a <= 0.25) {
    discard;
  }

  vec4 lighting_texture = vec4(texture(_t01, gl_FragCoord.xy / window_size).rgb, 1);
  vec3 base_lit = tex_color.rgb * lighting_ambient;
  vec3 additive_light = lighting_texture.rgb;
  tex_color.rgb = base_lit + additive_light;

  o_attachment0 = tex_color;
}