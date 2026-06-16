#version 420

layout(set = 0, binding = 0) uniform sampler2D _t00;

layout(push_constant) uniform push_constants_t {
  vec4 resolution_threshold_knee_mip;
  vec4 mode;
} pc;

layout(location = 0) in vec2 texture_coordinate;
layout(location = 0) out vec4 o_color;

vec3 apply_threshold(vec3 col) {
  float threshold = pc.resolution_threshold_knee_mip.z;
  float knee = pc.resolution_threshold_knee_mip.w;
  float brightness = max(col.r, max(col.g, col.b));

  if (threshold <= 0.0) {
    return col;
  }

  float rq = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
  rq = (rq * rq) / (4.0 * knee + 0.0001);
  float mask = max(rq, brightness - threshold) / max(brightness, 0.0001);
  return col * mask;
}

void main() {
  vec2 srcTexelSize = 1.0 / pc.resolution_threshold_knee_mip.xy;
  float x = srcTexelSize.x;
  float y = srcTexelSize.y;
  int mipLevel = int(pc.mode.x + 0.5);

  if (mipLevel == 0) {
    vec3 center = texture(_t00, texture_coordinate).rgb;
    vec3 a = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y + y)).rgb;
    vec3 b = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y + y)).rgb;
    vec3 c = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y - y)).rgb;
    vec3 d = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y - y)).rgb;

    center = apply_threshold(center);
    a = apply_threshold(a);
    b = apply_threshold(b);
    c = apply_threshold(c);
    d = apply_threshold(d);

    o_color = vec4((center + a + b + c + d) * 0.2, 1.0);
  }
  else {
    vec3 a = texture(_t00, vec2(texture_coordinate.x - 2.0 * x, texture_coordinate.y + 2.0 * y)).rgb;
    vec3 b = texture(_t00, vec2(texture_coordinate.x,           texture_coordinate.y + 2.0 * y)).rgb;
    vec3 c = texture(_t00, vec2(texture_coordinate.x + 2.0 * x, texture_coordinate.y + 2.0 * y)).rgb;

    vec3 d = texture(_t00, vec2(texture_coordinate.x - 2.0 * x, texture_coordinate.y)).rgb;
    vec3 e = texture(_t00, vec2(texture_coordinate.x,           texture_coordinate.y)).rgb;
    vec3 f = texture(_t00, vec2(texture_coordinate.x + 2.0 * x, texture_coordinate.y)).rgb;

    vec3 g = texture(_t00, vec2(texture_coordinate.x - 2.0 * x, texture_coordinate.y - 2.0 * y)).rgb;
    vec3 h = texture(_t00, vec2(texture_coordinate.x,           texture_coordinate.y - 2.0 * y)).rgb;
    vec3 i = texture(_t00, vec2(texture_coordinate.x + 2.0 * x, texture_coordinate.y - 2.0 * y)).rgb;

    vec3 j = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y + y)).rgb;
    vec3 k = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y + y)).rgb;
    vec3 l = texture(_t00, vec2(texture_coordinate.x - x, texture_coordinate.y - y)).rgb;
    vec3 m = texture(_t00, vec2(texture_coordinate.x + x, texture_coordinate.y - y)).rgb;

    vec3 col = e * 0.125;
    col += (a + c + g + i) * 0.03125;
    col += (b + d + f + h) * 0.0625;
    col += (j + k + l + m) * 0.125;

    o_color = vec4(max(col, 0.0001), 1.0);
  }
}
