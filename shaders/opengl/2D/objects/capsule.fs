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

float sd_capsule(vec2 p, vec2 a, vec2 b, float r) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

const float smoothing = 1.0/2.0;
const float outlineWidth = 3.0/16.0;
const float outerEdgeCenter = 0.5 - outlineWidth;

float cbrt(float x) {
  return pow(x, 1.0 / 3.0);
}

void main() {
  float distance = sd_capsule(frag_position.xy, instance_center0, instance_center1, instance_radius);
  float intensity = 1.0 - smoothstep(-1, 1, distance);
  float border = 1.0 - smoothstep(-cbrt(instance_radius*10) / 3 - smoothing, -cbrt(instance_radius*10) / 3 + smoothing, distance);
  vec4 color = vec4(mix(instance_outline_color.rgb, instance_color.rgb, border), intensity);
  color.a *= instance_color.a;
  o_attachment0 = color;
}