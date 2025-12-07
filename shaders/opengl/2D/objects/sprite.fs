#version 330

layout(location = 0) out vec4 o_attachment0;

in vec2 texture_coordinate;
in vec4 instance_color;
in vec3 frag_position;

flat in uint fs_flags;
flat in float object_seed;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform sampler2D _t02;
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
  vec4 tex_color_raw = texture(_t00, texture_coordinate);
  vec4 tex_color = tex_color_raw * instance_color;

  if (tex_color.a <= 0.25)
    discard;

  bool draw_mode = has_normal_map == 1 || has_specular_map == 1 || has_occlusion_map == 1;

  vec3 brightness_magic = vec3(0.2126, 0.7152, 0.0722);
  float tex_brightness = dot(tex_color_raw.rgb, brightness_magic);

  if (draw_mode == false) {
    vec3 light_color = texture(_t01, gl_FragCoord.xy / window_size).rgb;

    if (bool(fs_flags & 2u)) {
      float speed = 0.03;

      vec2 tc_noise  = texture_coordinate * 0.5 + vec2(0, -_time * speed);
      vec2 tc_noise2 = texture_coordinate * 0.1 + vec2(0, -_time * speed * 0.5);

      vec4 n1 = texture(_t05, tc_noise);
      vec4 n2 = texture(_t05, tc_noise2);
      vec4 nc = mix(n1, n2, 0.5);

      vec2 tc_offset =
        texture_coordinate +
        vec2(0, -_time * speed) +
        fract(vec2(object_seed, object_seed) * 5.324);

      tex_color_raw = texture(_t00, tc_offset + 0.6 * nc.rg);
      tex_color = tex_color_raw * instance_color;

      tex_brightness = dot(tex_color_raw.rgb, brightness_magic);
    }

    float lowest = min(min(light_color.r, light_color.g), light_color.b);

    vec3 mixed =
      tex_color_raw.rgb * lowest +
      tex_brightness * light_color * (1.0 - lowest);

    vec3 ambient = tex_color_raw.rgb * lighting_ambient;

    tex_color.rgb = (ambient + mixed) * instance_color.rgb;

    o_attachment0 = tex_color;
  }
  else {
    vec3 normal_world = vec3(0.0, 0.0, 1.0);

    if (has_normal_map == 1) {
      vec3 nt = texture(_t02, texture_coordinate).rgb * 2.0 - 1.0;
      nt.xy *= 0.5;
      nt = normalize(nt);

      vec3 n = vec3(0.0, 0.0, 1.0);
      vec3 t = vec3(1.0, 0.0, 0.0);
      vec3 b = vec3(0.0, 1.0, 0.0);

      normal_world = normalize(mat3(t, b, n) * nt);
    }

    float specular_intensity = 0.0;
    float roughness = 0.8;
    float metallic = 0.0;

    if (has_specular_map == 1) {
      vec3 s = texture(_t03, texture_coordinate).rgb;
      specular_intensity = max(max(s.r, s.g), s.b) * 2.0;
    }

    float occlusion = 1.0;
    if (has_occlusion_map == 1)
      occlusion = texture(_t04, texture_coordinate).r;

    vec3 light_map = texture(_t01, gl_FragCoord.xy / window_size).rgb;

    vec3 light_dir =
      length(light_map) > 0.1 ?
      normalize(vec3(light_map.r * 2.0 - 1.0, light_map.g * 2.0 - 1.0, 0.8)) :
      vec3(0.0, 0.0, 1.0);

    vec3 view_dir = vec3(0.0, 0.0, 1.0);

    float diff_raw = dot(normal_world, light_dir);
    float diff = max(diff_raw * 0.5 + 0.5, 0.2);

    float spec = 0.0;
    if (specular_intensity > 0.01) {
      float angle = max(dot(view_dir, reflect(-light_dir, normal_world)), 0.0);
      float shininess = mix(4.0, 64.0, 1.0 - roughness);
      spec = pow(angle, shininess);

      float fresnel = pow(1.0 - max(dot(normal_world, view_dir), 0.0), 2.0);
      spec *= mix(0.1, 1.0, fresnel);
    }

    float light_intensity = length(light_map);

    vec3 ambient  = lighting_ambient * tex_color.rgb * occlusion;
    vec3 diffuse  = diff * tex_color.rgb * light_map * 0.8;

    vec3 specular_color = mix(vec3(1.0), tex_color.rgb, metallic);
    vec3 specular = spec * specular_intensity * specular_color * light_intensity;

    o_attachment0 = vec4(ambient + diffuse + specular, tex_color.a);
  }
}