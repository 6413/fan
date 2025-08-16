#version 330

layout(location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;
in vec3 frag_position;

flat in uint fs_flags;
flat in float object_seed;

uniform sampler2D _t00; // texture
uniform sampler2D _t01; // light map
uniform sampler2D _t02; // normal map
uniform sampler2D _t03;
uniform sampler2D _t04;
uniform sampler2D _t05;
uniform vec3 lighting_ambient;
uniform vec2 window_size;

uniform float _time;

uniform int has_normal_map;
uniform int has_specular_map;
uniform int has_occlusion_map;


void main() {
  vec4 tex_color = texture(_t00, texture_coordinate) * instance_color;

  if (tex_color.a <= 0.25) {
    discard;
  }

  bool draw_mode = has_normal_map == 1 || has_specular_map == 1 || has_occlusion_map == 1;

  if (draw_mode == false) {
    vec4 lighting_texture = vec4(texture(_t01, gl_FragCoord.xy / window_size).rgb, 1);
    if (bool(fs_flags & 2u)) { // fatherload lava
      float speed = 0.03;
      vec2 tc_noise = texture_coordinate * 0.5 + vec2(0, -_time * speed);

      vec4 noise_col = texture(_t05, tc_noise);

      vec2 tc_noise2 = texture_coordinate * 0.1 + vec2(0, -_time * speed * 0.5);
      vec4 noise_col2 = texture(_t05, tc_noise2);

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
  else {
    vec3 normal_world = vec3(0.0, 0.0, 1.0);

    if (has_normal_map == 1) {
      vec3 normal_tangent = texture(_t02, texture_coordinate).rgb * 2.0 - 1.0;

      normal_tangent.xy *= 0.5;
      normal_tangent = normalize(normal_tangent);

      vec3 normal = vec3(0.0, 0.0, 1.0);
      vec3 tangent = vec3(1.0, 0.0, 0.0);
      vec3 bitangent = vec3(0.0, 1.0, 0.0);
      mat3 TBN = mat3(tangent, bitangent, normal);

      normal_world = normalize(TBN * normal_tangent);
    }

    float specular_intensity = 0.0;
    float roughness = 0.8;
    float metallic = 0.0;

    if (has_specular_map == 1) {
      vec3 specular_sample = texture(_t03, texture_coordinate).rgb;
      specular_intensity = max(specular_sample.r, max(specular_sample.g, specular_sample.b));
      specular_intensity *= 2.0;
    }

    float occlusion = 1.0;
    if (has_occlusion_map == 1) {
      occlusion = texture(_t04, texture_coordinate).r;
    }

    vec4 light_map = texture(_t01, gl_FragCoord.xy / window_size);

    vec3 light_dir;
    if (length(light_map.rgb) > 0.1) {
      light_dir = normalize(vec3(light_map.r * 2.0 - 1.0, light_map.g * 2.0 - 1.0, 0.8));
    }
    else {
      light_dir = normalize(vec3(0.0, 0.0, 1.0));
    }

    vec3 view_dir = normalize(vec3(0.0, 0.0, 1.0));

    // prevent completely dark areas
    float diff_raw = dot(normal_world, light_dir);
    float diff = max(diff_raw * 0.5 + 0.5, 0.2);

    float spec = 0.0;
    if (specular_intensity > 0.01) {
      vec3 reflect_dir = reflect(-light_dir, normal_world);
      float spec_angle = max(dot(view_dir, reflect_dir), 0.0);

      // Blinn-Phong (broken)
      // vec3 halfway_dir = normalize(light_dir + view_dir);
      // float spec_angle = max(dot(normal_world, halfway_dir), 0.0);

      float shininess = mix(4.0, 64.0, 1.0 - roughness);
      spec = pow(spec_angle, shininess);

      float fresnel = pow(1.0 - max(dot(normal_world, view_dir), 0.0), 2.0);
      spec *= mix(0.1, 1.0, fresnel);
    }

    float light_intensity = length(light_map.rgb);

    vec3 ambient = lighting_ambient * tex_color.rgb * occlusion;
    vec3 diffuse = diff * tex_color.rgb * light_map.rgb * 0.8;

    vec3 specular_color = mix(vec3(1.0), tex_color.rgb, metallic);
    vec3 specular = spec * specular_intensity * specular_color * light_intensity;

    vec3 final_color = ambient + diffuse + specular;

    o_attachment0 = vec4(final_color, tex_color.a);
  }
}