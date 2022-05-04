R"(
#version 330

in vec2 texture_coordinate;

in vec4 i_color;
in vec2 fragment_position;

out vec4 o_color;

uniform sampler2D texture_sampler;

vec4 calculate_lighting(vec3 ambient, vec2 light_position, vec3 light_color, float intensity, float radius)
{
  if (intensity < 0.1 || radius < 0.1) {
    return vec4(0, 0, 0, 1.0);
  }

  vec2 light_direction = normalize(light_position - fragment_position);
  float diff = max(length(light_direction), 0.0);
  vec3 diffuse = diff * light_color;

  float distance = length(light_position - fragment_position);
  float cutoff = radius / 2;
  if (distance > radius) {
    distance *= distance / radius;
  }
  float distance_strength = distance / intensity;
  float attenuation = 1.0 / (distance_strength * distance_strength);
  
  ambient *= attenuation;
  diffuse *= attenuation;
  diffuse += clamp(1.0 / ((distance / (radius)* (distance / (radius)))), 0, 0.5);

  return vec4(ambient + diffuse, 1.0);
}

vec4 calculate_lighting1(vec3 ambient, vec2 light_position, vec3 light_color, float intensity, float radius)
{
  vec2 light_direction = normalize(light_position - fragment_position);
  float diff = max(length(light_direction), 0.0);
  vec3 diffuse = diff * light_color;

  float distance = length(light_position - fragment_position);
  if (fragment_position.y > 0) {
    distance *= (fragment_position.y * fragment_position.y) / 100000 + 1;
  }
  float distance_strength = distance / intensity;
  float attenuation = 1.0 / (distance_strength * distance_strength);
  
  ambient *= attenuation;
  diffuse *= attenuation;
  diffuse += clamp(1.0 / ((distance / (radius / 2)* (distance / (radius / 2)))), 0, 0.5);

  return vec4(ambient + diffuse, 1.0);
}


void main() {

  vec2 flipped_y = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);

  vec4 texture_color = texture(texture_sampler, flipped_y);

  o_color = texture_color * i_color;
}
)"