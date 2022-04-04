R"(
#version 130

in vec2 texture_coordinate;

in vec4 i_color;
in vec2 f_position;
in float allow_lighting;

out vec4 o_color;

uniform sampler2D texture_sampler;

/*
  0-1 position
  2 radius
  3 intensity
  4 ambient strength
  5-7 color
  
*/
uniform float render_codef[255];
uniform uint render_codeu[4];

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

          vec4 ambient = vec4(ambient_strength * light_color, 1);

          vec2 d = light_position - f_position;
          float diff = max(1.0 - (abs(length(d)) / radius), 0);
          diff *= intensity;

          vec4 diffuse = vec4(light_color * diff, 1);

          o_color += (diffuse) * texture_color;

          break;
         }
      }
      render_code_index++;
    }
    o_color /= render_code_n;
    o_color *= i_color;
  }
}
)"