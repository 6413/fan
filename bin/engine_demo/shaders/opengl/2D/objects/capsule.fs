#version 330

layout (location = 0) out vec4 o_attachment0;

in vec4 instance_color;
in vec2 instance_center0;
in vec2 instance_center1;
in float instance_radius;
in vec3 frag_position;
in vec2 texture_coordinate;
in vec4 instance_outline_color;
flat in uint flags;

float smoothing = 2.0;

uniform float camera_zoom;

float sd_capsule(vec2 p, vec2 a, vec2 b, float r) {
  vec2 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

float cbrt(float x) {
  return pow(x, 1.0 / 3.0);
}

void main() {
  smoothing /= camera_zoom;

  float distance = sd_capsule(frag_position.xy, instance_center0, instance_center1, instance_radius);
  float intensity = smoothing > 0.0
    ? 1.0 - smoothstep(-smoothing, smoothing, distance)
    : (distance < 0.0 ? 1.0 : 0.0);

  float edge = -cbrt(instance_radius * 10.0) / 3.0;
  float border = smoothing > 0.0
    ? 1.0 - smoothstep(edge - smoothing, edge + smoothing, distance)
    : (distance < edge ? 1.0 : 0.0);

  vec4 color = vec4(mix(instance_outline_color.rgb, instance_color.rgb, border), intensity);
  color.a *= instance_color.a;

  o_attachment0 = color;
}