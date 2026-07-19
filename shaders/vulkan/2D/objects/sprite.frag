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
  float time;
  uint lightmap_id;
  uint pad2;
  vec4 lighting_ambient;
} constants;

layout(set = 0, binding = 2) uniform sampler2D textures[1024];

void main() {
  vec4 color = texture(textures[constants.texture_id], texture_coordinate) * instance_color;
  
  vec2 lightmap_size = vec2(textureSize(textures[constants.lightmap_id], 0));
  vec2 lightmap_uv = gl_FragCoord.xy / lightmap_size;
  vec3 light = texture(textures[constants.lightmap_id], lightmap_uv).rgb;
  
  o_attachment0 = color;
  o_attachment0.rgb *= (constants.lighting_ambient.rgb + light);
}