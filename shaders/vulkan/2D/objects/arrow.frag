#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 arrow_coordinate;
layout(location = 2) in flat vec4 arrow_params;
layout(location = 3) in flat float arrow_half_size;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
} constants;

layout(set = 0, binding = 2) uniform sampler2D textures[1024];

float arrow_sdf(vec2 p, float h, float shaft_thickness) {
  float head_base = -0.15 * h;
  float head_half_width = 0.95 * h;
  float slope = head_half_width / (h - head_base);

  float d_edge = (abs(p.y) - slope * (h - p.x)) / sqrt(1.0 + slope * slope);
  float d_head = max(d_edge, head_base - p.x);

  float shaft_end = head_base + 0.1 * h;
  float shaft_center = (-h + shaft_end) * 0.5;
  float shaft_half_length = (shaft_end + h) * 0.5;
  float d_shaft = max(abs(p.x - shaft_center) - shaft_half_length, abs(p.y) - shaft_thickness * 0.5);

  return min(d_head, d_shaft);
}

void main() {
  float segment_length = arrow_params.x;
  float spacing = arrow_params.y;
  float shaft_thickness = arrow_params.z;
  int flags = int(arrow_params.w + 0.5);
  bool hide_end = (flags & 1) != 0;
  bool hide_start = (flags & 2) != 0;

  float cell = floor(arrow_coordinate.x / spacing + 0.5);
  float center = cell * spacing;
  vec2 p = vec2(arrow_coordinate.x - center, arrow_coordinate.y);

  float d = arrow_sdf(p, arrow_half_size, shaft_thickness);
  float w = max(fwidth(d), 1e-4);
  float alpha = clamp(0.5 - d / w, 0.0, 1.0);

  bool inside_segment = center >= -0.5 && center <= segment_length + 0.5;
  bool hidden_end = hide_end && center >= segment_length - 0.5;
  bool hidden_start = hide_start && center <= 0.5;
  if (!inside_segment || hidden_end || hidden_start) {
    alpha = 0.0;
  }
  if (alpha <= 0.0) {
    discard;
  }
  o_attachment0 = vec4(instance_color.rgb, instance_color.a * alpha);
}
