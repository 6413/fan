R"(
#version 130

in vec4 instance_color;
in vec2 instance_position;
in vec2 instance_fragment_position;
in float instance_radius;

out vec4 color;

void main() {
  float distance = distance(gl_FragCoord.xy, instance_position);
  if (distance < instance_radius) {
    const float smoothness = 2;
    float a = smoothstep(0, 1, (instance_radius - distance) / smoothness);
    color = vec4(instance_color.rgb, a * instance_color.w);
  }
}
)"