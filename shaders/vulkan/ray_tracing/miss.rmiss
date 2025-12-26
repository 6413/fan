#version 460
#extension GL_EXT_ray_tracing : require

struct Payload {
  vec3 color;
  vec3 normal;
  vec2 uv;
  uint material_id;
  float ao;
  int depth;
};
layout(location = 0) rayPayloadInEXT Payload payload;

void main(){
  vec3 d = normalize(gl_WorldRayDirectionEXT);
  float t = 0.5*(d.y+1.0);
  payload.color = mix(vec3(0.1,0.15,0.2), vec3(0.6,0.7,0.9), t);
}