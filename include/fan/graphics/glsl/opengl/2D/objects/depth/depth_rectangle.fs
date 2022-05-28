R"(
#version 130

in vec4 color;

out vec4 r_color;

in vec2 f_position;

in vec2 light_position;
in vec4 light_color;
in float light_brightness;
in float light_angle;

uniform bool enable_lighting;

uniform float world_light_strength;

in float render_depth;

void main() {
  r_color = color;
  gl_FragDepth = 0.999 - render_depth;
}
)"