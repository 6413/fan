#version 330

layout (location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;
in vec3 frag_position;

flat in uint fs_flags;
flat in float object_seed;

uniform sampler2D _t00; // texture
uniform sampler2D _t01; // light map
uniform sampler2D _t02; // normal map
uniform int draw_mode = 0;
uniform vec3 lighting_ambient;
uniform vec2 window_size;

uniform float _time;

void main() {
  vec4 tex_color = texture(_t00, texture_coordinate) * instance_color;

  if (tex_color.a <= 0.25) {
      discard;
  }

  if (draw_mode == 0) {
    vec4 lighting_texture = vec4(texture(_t01, gl_FragCoord.xy / window_size).rgb, 1);
		if (bool(fs_flags & 2u)) { // fatherload lava
			float speed = 0.03;
			vec2 tc_noise = texture_coordinate * 0.5 + vec2(0, -_time * speed);

			vec4 noise_col = texture(_t02, tc_noise);

			vec2 tc_noise2 = texture_coordinate * 0.1 + vec2(0, -_time * speed * 0.5);
			vec4 noise_col2 = texture(_t02, tc_noise2);

			vec4 noise_combined = mix(noise_col, noise_col2, 0.5);

			vec2 tc_offset = texture_coordinate + vec2(0, -_time * speed) + fract(vec2(object_seed, object_seed) * 5.324);

			tex_color = texture(_t00, tc_offset + 0.6 * noise_combined.rg) * instance_color;
      tex_color.rgb *= lighting_ambient + lighting_texture.rgb;
		}
    else if (bool(fs_flags & 4u)) { // additive
      vec3 base_lit = tex_color.rgb * lighting_ambient;
      vec3 additive_light = lighting_texture.rgb;
      tex_color.rgb = base_lit + additive_light;
    }
    else if (bool(fs_flags & 8u)) { // multiplicative
      tex_color.rgb *= lighting_ambient + lighting_texture.rgb;
    }

    o_attachment0 = tex_color;
  }
  else if (draw_mode == 1) { // normal calculation
    vec3 normal_tangent = texture(_t02, texture_coordinate).rgb * 2.0 - 1.0;
    vec3 normal_world = normalize(normal_tangent);
    vec3 corrected_frag_coord = vec3(gl_FragCoord.x, window_size.y - gl_FragCoord.y, gl_FragCoord.z);
    vec4 light_map = texture(_t01, gl_FragCoord.xy / window_size);
    vec3 light_dir = normalize(vec3(light_map.r * 2.0 - 1.0, light_map.g * 2.0 - 1.0, 1.0));
    float diff = max(dot(normal_world, light_dir), 0.0);
    vec3 normal_shading = normal_world * 0.5 + 0.5;

    //vec3 ambient = lighting_ambient * tex_color.rgb * normal_shading;
    //vec3 lighting = ambient + diff * tex_color.rgb * light_map.rgb;
    //o_attachment0 = vec4(lighting, tex_color.a);

    vec3 ambient = lighting_ambient * tex_color.rgb;
    
    vec3 normal_influence = mix(vec3(1.0), normal_shading, 0.5);
    vec3 ambient_with_normals = ambient * normal_influence;
    
    vec3 diffuse = diff * tex_color.rgb * light_map.rgb;
    vec3 lighting = ambient_with_normals + diffuse;
    o_attachment0 = vec4(lighting, tex_color.a);
  }
}
