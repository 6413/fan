#version 330

layout(location = 1) out vec4 o_attachment1;
//layout (location = 3) out vec4 o_attachment3;
uniform float multiplier = 1.17;
uniform float multiplier2 = 0.430;

uniform vec3 lighting_ambient;

in vec4 instance_color;
in vec3 instance_position;
in vec2 instance_size;
in vec3 frag_position;

in vec2 uv;

flat in uint fs_flags;

void main() {
  vec4 lightColor = vec4(0.0, 0.0, 0.0, 1.0);
  float distance = length(frag_position - instance_position);
  float radius = instance_size.x;
  float smooth_edge = radius;
  float intensity;

  if (fs_flags == 0u) {
  // Circle lighting
    intensity = 1.0 - length(uv);
  }
  else if (fs_flags == 1u) {
    // Square lighting
    smooth_edge *= multiplier;
    radius *= multiplier2;
    vec3 diff = abs(frag_position - instance_position) - vec3(radius);
    float edge_distance = max(diff.x, diff.y);
    intensity = 1.0 - smoothstep(radius - smooth_edge, radius, edge_distance);
  }
  else if (fs_flags == 2u) {
    if (abs(frag_position.x - instance_position.x) < radius / 3 && abs(frag_position.y - instance_position.y) < radius / 3) {
      intensity = 1.0;
    }
    else {
      smooth_edge *= multiplier;
      radius *= multiplier2;
      vec3 diff = abs(frag_position - instance_position) - vec3(radius);
      float edge_distance = max(diff.x, diff.y);
      intensity = 1.0 - smoothstep(radius - smooth_edge, radius, edge_distance);
    }
  }
  else if (fs_flags == 3u) {
    vec3 lightDir = vec3(-1.0, 0.0, 0.0);
    vec3 pixelDir = normalize(frag_position - instance_position);
    float angle = max(dot(lightDir, pixelDir), 0.0);
    intensity = smoothstep(0.8, 1.0, angle);
    intensity *= 1.0 - smoothstep(radius / 2, radius, distance);
  }
  else if (fs_flags == 4u) {
    vec3 lightDir = vec3(1.0, 0.0, 0.0);
    vec3 pixelDir = normalize(frag_position - instance_position);
    float angle = max(dot(lightDir, pixelDir), 0.0);
    intensity = smoothstep(0.8, 1.0, angle);
    intensity *= 1.0 - smoothstep(radius / 2, radius, distance);
  }
  else if (fs_flags == 5u) {
    vec3 lightDir = vec3(0.0, -1.0, 0.0);
    vec3 pixelDir = normalize(frag_position - instance_position);
    float angle = max(dot(lightDir, pixelDir), 0.0);
    intensity = smoothstep(0.8, 1.0, angle);
    intensity *= 1.0 - smoothstep(radius / 2, radius, distance);
  }
  else if (fs_flags == 6u) {
    vec3 lightDir = vec3(0.0, 1.0, 0.0);
    vec3 pixelDir = normalize(frag_position - instance_position);
    float angle = max(dot(lightDir, pixelDir), 0.0);
    intensity = smoothstep(0.8, 1.0, angle);
    intensity *= 1.0 - smoothstep(radius / 2, radius, distance);
  }


  lightColor = instance_color * intensity;
  o_attachment1 = lightColor;
}
