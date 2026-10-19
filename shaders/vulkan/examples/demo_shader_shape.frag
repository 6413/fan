#version 450
layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint texture_id;
  uint camera_id;
  uint texture_id1;
  uint texture_id2;
  uint texture_id3;
} constants;

layout(set = 0, binding = 2) uniform sampler2D textures[1024];

void main() {
  vec4 tex_color = texture(textures[constants.texture_id], texture_coordinate) * instance_color;
  if (tex_color.a <= 0.25) {
    discard;
  }
  float luminance = dot(tex_color.rgb, vec3(0.299, 0.587, 0.114));
  tex_color.rgb = instance_color.rgb * luminance;
  tex_color.a *= instance_color.a;
  o_attachment0 = tex_color;
}