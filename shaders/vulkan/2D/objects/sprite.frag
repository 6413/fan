#version 450

layout(location = 0) in vec4 instance_color;
layout(location = 1) in vec2 texture_coordinate;
layout(location = 2) in flat uvec4 instance_texture_ids;
layout(location = 3) in flat uint fs_flags;
layout(location = 4) in flat float object_seed;
layout(location = 0) out vec4 o_attachment0;

layout(push_constant) uniform constants_t {
  uint _pad0;
  uint camera_id;
  uint default_texture_id;
  uint _pad2;
  uint _pad3;
  float time;
  uint lightmap_id;
  float ambient_floor;
  vec4 lighting_ambient;
} constants;

layout(set = 0, binding = 2) uniform sampler2D textures[1024];

vec3 rgb_to_hsl(vec3 rgb) {
  float maxc = max(max(rgb.r, rgb.g), rgb.b);
  float minc = min(min(rgb.r, rgb.g), rgb.b);
  float delta = maxc - minc;
  float l = (maxc + minc) * 0.5;
  if (delta < 0.00001) return vec3(0.0, 0.0, l);
  float s = delta / (1.0 - abs(2.0 * l - 1.0));
  float h;
  if (maxc == rgb.r) { h = (rgb.g - rgb.b) / delta; if (h < 0.0) h += 6.0; }
  else if (maxc == rgb.g) { h = ((rgb.b - rgb.r) / delta) + 2.0; }
  else { h = ((rgb.r - rgb.g) / delta) + 4.0; }
  return vec3(h / 6.0, s, l);
}

vec3 hsl_to_rgb(vec3 hsl) {
  float h = hsl.x, s = hsl.y, l = hsl.z;
  if (s < 0.00001) return vec3(l);
  float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
  float p = 2.0 * l - q;
  vec3 t = vec3(h + 1.0/3.0, h, h - 1.0/3.0);
  t = fract(t);
  vec3 rgb;
  for (int i = 0; i < 3; i++) {
    if (t[i] < 1.0/6.0) rgb[i] = p + (q - p) * 6.0 * t[i];
    else if (t[i] < 0.5) rgb[i] = q;
    else if (t[i] < 2.0/3.0) rgb[i] = p + (q - p) * (2.0/3.0 - t[i]) * 6.0;
    else rgb[i] = p;
  }
  return rgb;
}

void main() {
  vec4 tex_color_raw = texture(textures[instance_texture_ids.x], texture_coordinate);
  vec4 tex_color;

  bool use_hsl = bool(fs_flags & 16u);
  if (use_hsl) {
    vec3 hsl = rgb_to_hsl(tex_color_raw.rgb);
    hsl.x = fract(hsl.x + instance_color.r / 360.0);
    hsl.y = instance_color.g < 0.0 ? hsl.y * (1.0 + instance_color.g / 100.0) : hsl.y + (1.0 - hsl.y) * (instance_color.g / 100.0);
    hsl.y = clamp(hsl.y, 0.0, 1.0);
    hsl.z = instance_color.b < 0.0 ? hsl.z * (1.0 + instance_color.b / 100.0) : hsl.z + (1.0 - hsl.z) * (instance_color.b / 100.0);
    hsl.z = clamp(hsl.z, 0.0, 1.0);
    tex_color_raw.rgb = hsl_to_rgb(hsl);
    tex_color = vec4(tex_color_raw.rgb, tex_color_raw.a * instance_color.a);
  } else {
    tex_color = tex_color_raw * instance_color;
  }

  vec2 viewport_size = vec2(textureSize(textures[constants.lightmap_id], 0));
  vec2 lightmap_uv = gl_FragCoord.xy / viewport_size;
  vec3 light_color = texture(textures[constants.lightmap_id], lightmap_uv).rgb;

  bool has_normal   = instance_texture_ids.y != constants.default_texture_id;
  bool has_specular = instance_texture_ids.z != constants.default_texture_id;
  bool has_occlusion = instance_texture_ids.w != constants.default_texture_id;
  bool draw_mode = has_normal || has_specular || has_occlusion;

  if (draw_mode) {
    vec3 normal_world = vec3(0.0, 0.0, 1.0);
    if (has_normal) {
      vec3 nt = texture(textures[instance_texture_ids.y], texture_coordinate).rgb * 2.0 - 1.0;
      nt.xy *= 0.5;
      nt = normalize(nt);
      vec3 n = vec3(0.0, 0.0, 1.0);
      vec3 t = vec3(1.0, 0.0, 0.0);
      vec3 b = vec3(0.0, 1.0, 0.0);
      normal_world = normalize(mat3(t, b, n) * nt);
    }

    float specular_intensity = 0.0;
    float roughness = 0.8;
    if (has_specular) {
      vec3 s = texture(textures[instance_texture_ids.z], texture_coordinate).rgb;
      specular_intensity = max(max(s.r, s.g), s.b) * 2.0;
    }

    float occlusion = 1.0;
    if (has_occlusion) occlusion = texture(textures[instance_texture_ids.w], texture_coordinate).r;

    vec3 light_dir = length(light_color) > 0.1
      ? normalize(vec3(light_color.r * 2.0 - 1.0, light_color.g * 2.0 - 1.0, 0.8))
      : vec3(0.0, 0.0, 1.0);
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

    float light_intensity = length(light_color);
    vec3 ambient  = constants.lighting_ambient.rgb * tex_color.rgb * occlusion;
    vec3 diffuse  = diff * tex_color.rgb * light_color * 0.8;
    vec3 specular_color = mix(vec3(1.0), tex_color.rgb, 0.0);
    vec3 specular = spec * specular_intensity * specular_color * light_intensity;
    o_attachment0 = vec4(ambient + diffuse + specular, tex_color.a);
  } else {
    vec3 brightness_magic = vec3(0.2126, 0.7152, 0.0722);
    float tex_brightness = dot(tex_color_raw.rgb, brightness_magic);
    float lowest = min(min(light_color.r, light_color.g), light_color.b);
    vec3 mixed = tex_color_raw.rgb * lowest + tex_brightness * light_color * (1.0 - lowest);
    vec3 ambient = tex_color_raw.rgb * constants.lighting_ambient.rgb;
    if (use_hsl) {
      tex_color.rgb = ambient + mixed;
    } else {
      tex_color.rgb = (ambient + mixed) * instance_color.rgb;
    }
    o_attachment0 = tex_color;
  }
}
