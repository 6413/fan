#version 120

varying vec4 instance_color;

void main() {
  gl_FragColor = instance_color;
}
