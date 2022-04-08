R"(
#version 130

in vec2 texture_coordinate;

in vec4 i_color;
in vec2 fragment_position;
in float allow_lighting;

out vec4 o_color;

uniform sampler2D texture_sampler;

/*
  0-1 position
  2 radius
  3 intensity
  4 ambient strength
  5-7 color
  8 angle
  9-10 rotation point
  
*/
uniform float render_codef[255];
uniform uint render_codeu[4];

bool gamma = true;

vec4 calculate_lighting(vec3 ambient, vec2 light_position, vec3 light_color, float intensity, float radius)
{
  vec2 light_direction = normalize(light_position - fragment_position);
  float diff = max(length(light_direction), 0.0);
  vec3 diffuse = diff * light_color;

  float distance = length(light_position - fragment_position);
  float distance_strength = distance / intensity;
  float attenuation = 1.0 / (gamma ? distance_strength * distance_strength : distance_strength);
  
  ambient *= attenuation;
  diffuse *= attenuation;
  diffuse += clamp(1.0 / ((distance / (radius / 2)* (distance / (radius / 2)))), 0, 0.5);

  return vec4(ambient + diffuse, 1.0);
}

void main() {

  vec2 flipped_y = vec2(texture_coordinate.x, 1.0 - texture_coordinate.y);

  vec4 texture_color = texture(texture_sampler, flipped_y);

  uint render_code_n = render_codeu[0] & 0xffu;
  uint render_code_index = 0u;
  uint render_codeu_byte_index = 1u;
  uint render_codef_index = 0u;
  if (allow_lighting == 0 || render_code_n == 0u) {
    o_color = texture_color * i_color;
  }
  else {
    o_color = vec4(0, 0, 0, 0);
    while (render_code_index != render_code_n) {
      uint render_code = /*render_codeu[render_codeu_byte_index / 4u] >> (render_codeu_byte_index % 4u)*/ 0u;
      switch(render_code) {
         case 0u: {
          vec2 light_position;
          light_position.x = render_codef[render_codef_index++];
          light_position.y = render_codef[render_codef_index++];

          float radius = render_codef[render_codef_index++];
          float intensity = render_codef[render_codef_index++];
          float ambient_strength = render_codef[render_codef_index++];

          vec3 light_color = vec3(
            render_codef[render_codef_index++], 
            render_codef[render_codef_index++], 
            render_codef[render_codef_index++]
          );

          vec2 rotation_point = vec2(
            render_codef[render_codef_index++], 
            render_codef[render_codef_index++]
          );

          float angle = render_codef[render_codef_index++];

          float cosine = cos(-angle);
          float sine = sin(-angle);
          vec2 rotate = rotation_point;
          light_position.x += rotate.x * cosine - rotate.y * sine;
          light_position.y += rotate.x * sine + rotate.y * cosine;

          light_position.x -= rotation_point.x;
          light_position.y -= rotation_point.y;

          vec3 ambient = ambient_strength * light_color;

          o_color += calculate_lighting(ambient, light_position, light_color, intensity, radius);

          break;
         }
      }
      render_code_index++;
    }
    o_color *= (texture_color * i_color);
    o_color.a = texture_color.a;
  }
}
)"