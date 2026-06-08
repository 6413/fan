#version 330 core
in vec4 instance_color;
in vec2 line_start;
in vec2 line_end;
in float line_radius;
in vec3 frag_position;
in vec2 texture_coordinate;
in vec2 ba;
in float ba_len2;
uniform float camera_zoom;
out vec4 color;

float sd_capsule(vec2 p, vec2 a, float r) {
  vec2 pa = p - a;
  float h = clamp(dot(pa, ba) / ba_len2, 0.0, 1.0);
  return length(pa - ba * h) - r;
}

void main() {
  float distance = sd_capsule(frag_position.xy, line_start, line_radius);
  float smoothing = max(fwidth(distance), 0.5 / camera_zoom);
  float alpha = 1.0 - smoothstep(-smoothing, smoothing, distance);
  color = vec4(instance_color.rgb, alpha * instance_color.a);
}