#version 120

varying vec2 texture_coordinate;
varying vec4 instance_color;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;
uniform vec3 lighting_ambient;
uniform vec2 window_size;
uniform float _time;
uniform vec2 offset;

void main() {
  vec4 tex_color = texture2D(_t00, texture_coordinate) * instance_color;
  if (tex_color.a <= 0.25) {
    discard;
  }
  gl_FragColor = tex_color;
}