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
vec3 rgb_to_hsl(vec3 rgb) {
  float maxc = max(max(rgb.r, rgb.g), rgb.b);
  float minc = min(min(rgb.r, rgb.g), rgb.b);
  float delta = maxc - minc;
  float l = (maxc + minc) * 0.5;
  if (delta < 0.00001) {
    return vec3(0.0, 0.0, l);
  }
  float s = delta / (1.0 - abs(2.0 * l - 1.0));
  float h;
  if (maxc == rgb.r) {
    h = (rgb.g - rgb.b) / delta;
    if (h < 0.0) h += 6.0;
  }
  else if (maxc == rgb.g) {
    h = ((rgb.b - rgb.r) / delta) + 2.0;
  }
  else {
    h = ((rgb.r - rgb.g) / delta) + 4.0;
  }
  return vec3(h / 6.0, s, l);
}
vec3 hsl_to_rgb(vec3 hsl) {
  float h = hsl.x;
  float s = hsl.y;
  float l = hsl.z;
  if (s < 0.00001) {
    return vec3(l);
  }
  float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
  float p = 2.0 * l - q;
  vec3 t = vec3(h + 1.0/3.0, h, h - 1.0/3.0);
  t = fract(t);
  vec3 rgb;
  for (int i = 0; i < 3; i++) {
    if (t[i] < 1.0/6.0) {
      rgb[i] = p + (q - p) * 6.0 * t[i];
    }
    else if (t[i] < 0.5) {
      rgb[i] = q;
    }
    else if (t[i] < 2.0/3.0) {
      rgb[i] = p + (q - p) * (2.0/3.0 - t[i]) * 6.0;
    }
    else {
      rgb[i] = p;
    }
  }
  return rgb;
}
void main() {
  vec4 tex_color_raw = texture(_t00, texture_coordinate);
  vec4 tex_color;
  if (bool(fs_flags & 16u)) {
    float hue_shift = instance_color.r;
    float sat_adjust = instance_color.g;
    float light_adjust = instance_color.b;
    vec3 hsl = rgb_to_hsl(tex_color_raw.rgb);
    hsl.x = fract(hsl.x + hue_shift / 360.0);
    if (sat_adjust < 0.0) {
      hsl.y = hsl.y * (1.0 + sat_adjust / 100.0);
    }
    else {
      hsl.y = hsl.y + (1.0 - hsl.y) * (sat_adjust / 100.0);
    }
    hsl.y = clamp(hsl.y, 0.0, 1.0);
    if (light_adjust < 0.0) {
      hsl.z = hsl.z * (1.0 + light_adjust / 100.0);
    }
    else {
      hsl.z = hsl.z + (1.0 - hsl.z) * (light_adjust / 100.0);
    }
    hsl.z = clamp(hsl.z, 0.0, 1.0);
    tex_color_raw.rgb = hsl_to_rgb(hsl);
    tex_color = vec4(tex_color_raw.rgb, tex_color_raw.a * instance_color.a);
  }
  else {
    tex_color = tex_color_raw * instance_color;
  }
  if (tex_color.a <= 0.25)
    discard;
  bool draw_mode = has_normal_map == 1 || has_specular_map == 1 || has_occlusion_map == 1;
  vec3 brightness_magic = vec3(0.2126, 0.7152, 0.0722);
  float tex_brightness = dot(tex_color_raw.rgb, brightness_magic);
  if (draw_mode == false) {
    vec3 light_color = texture(_t01, gl_FragCoord.xy / window_size).rgb;
    /*
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
    */
    float lowest = min(min(light_color.r, light_color.g), light_color.b);
    vec3 mixed =
      tex_color_raw.rgb * lowest +
      tex_brightness * light_color * (1.0 - lowest);
    vec3 ambient = tex_color_raw.rgb * lighting_ambient;
    if (bool(fs_flags & 16u)) {
      tex_color.rgb = ambient + mixed;
    }
    else {
      tex_color.rgb = (ambient + mixed) * instance_color.rgb;
    }
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