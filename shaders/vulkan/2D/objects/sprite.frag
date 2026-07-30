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
  if (delta < 0.00001) {
    return vec3(0.0, 0.0, l);
  }
  float s = delta / (1.0 - abs(2.0 * l - 1.0));
  float h;
  if (maxc == rgb.r) {
    h = (rgb.g - rgb.b) / delta;
    if (h < 0.0) {
      h += 6.0;
    }
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

  vec3 t = fract(vec3(h + 1.0 / 3.0, h, h - 1.0 / 3.0));

  vec3 rgb;

  for (int i = 0; i < 3; i++) {
    if (t[i] < 1.0 / 6.0) {
      rgb[i] = p + (q - p) * 6.0 * t[i];
    }
    else if (t[i] < 0.5) {
      rgb[i] = q;
    }
    else if (t[i] < 2.0 / 3.0) {
      rgb[i] = p + (q - p) * (2.0 / 3.0 - t[i]) * 6.0;
    }
    else {
      rgb[i] = p;
    }
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

    hsl.y = instance_color.g < 0.0
      ? hsl.y * (1.0 + instance_color.g / 100.0)
      : hsl.y + (1.0 - hsl.y) * (instance_color.g / 100.0);

    hsl.z = instance_color.b < 0.0
      ? hsl.z * (1.0 + instance_color.b / 100.0)
      : hsl.z + (1.0 - hsl.z) * (instance_color.b / 100.0);

    hsl.y = clamp(hsl.y, 0.0, 1.0);
    hsl.z = clamp(hsl.z, 0.0, 1.0);

    tex_color_raw.rgb = hsl_to_rgb(hsl);
    tex_color = vec4(tex_color_raw.rgb, tex_color_raw.a * instance_color.a);
  }
  else {
    tex_color = tex_color_raw * instance_color;
  }

  vec2 viewport_size = vec2(textureSize(textures[constants.lightmap_id], 0));
  vec2 lightmap_uv = gl_FragCoord.xy / viewport_size;

  vec3 light = texture(textures[constants.lightmap_id], lightmap_uv).rgb;

  light = pow(light, vec3(2.0));

  vec3 ambient = tex_color_raw.rgb * constants.lighting_ambient.rgb;

  vec3 color = tex_color_raw.rgb * light;

  color += ambient;

  color = color / (1.0 + color);

  if (!use_hsl) {
    color *= instance_color.rgb;
  }

  o_attachment0 = vec4(color, tex_color.a);
}