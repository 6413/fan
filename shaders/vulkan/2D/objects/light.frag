#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec3 instance_position;
layout(location = 2) in vec2 instance_size;
layout(location = 3) in vec3 frag_position;
layout(location = 4) in vec2 uv;
layout(location = 5) flat in uint fs_flags;
layout(location = 0) out vec4 o_attachment0;

const vec2 magic_expand = vec2(1.0f);

void main() {
  float intensity = 0.0;
  float radius = max(instance_size.x * magic_expand.x, 0.0001);
  
  if (fs_flags == 0u) {
   float dist_sq = dot(uv, uv);
    
    float core_size = 0.05; 
    float spread = 0.2; 
    float attenuation = core_size / (dist_sq * spread + core_size);
    
    float window = max(1.0 - dist_sq, 0.0);
    intensity = attenuation * window;
  }
  else if (fs_flags == 1u) {
    vec2 half_size = instance_size * magic_expand;
    vec2 d = abs(frag_position.xy - instance_position.xy);
    d.y *= half_size.x / max(half_size.y, 0.0001);
    intensity = 1.0 - smoothstep(0.0, 1.0, max(d.x, d.y) / half_size.x);
  }
  else if (fs_flags == 2u) {
    float smooth_edge = radius * 1.17;
    float adjusted_radius = radius * 0.430;
    vec2 diff = abs(frag_position.xy - instance_position.xy) - vec2(adjusted_radius);
    float edge_distance = length(max(diff, 0.0)) + min(max(diff.x, diff.y), 0.0);
    intensity = edge_distance <= 0.0 ? 1.0 : 1.0 - smoothstep(0.0, smooth_edge, edge_distance);
  }
  else if (fs_flags >= 3u && fs_flags <= 6u) {
    const vec2 dirs[4] = vec2[](
      vec2(-1.0, 0.0), vec2(1.0, 0.0), 
      vec2(0.0, -1.0), vec2(0.0, 1.0)
    );
    vec2 pixelDir = normalize(frag_position.xy - instance_position.xy);
    float angle = max(dot(dirs[fs_flags - 3u], pixelDir), 0.0);
    float dist = length(frag_position.xy - instance_position.xy);
    intensity = smoothstep(0.8, 1.0, angle) * (1.0 - smoothstep(radius * 0.5, radius, dist));
  }
  
  o_attachment0 = instance_color * intensity;
}