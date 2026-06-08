#version 120

varying vec2 tex_coord;
varying vec3 v_normal;
varying vec3 v_pos;
varying vec4 bw;

varying vec4 vcolor;

varying vec3 c_tangent;
varying vec3 c_bitangent;
//layout (location = 0) out vec4 color;

uniform sampler2D _t00; // aiTextureType_NONE
uniform sampler2D _t01; // aiTextureType_DIFFUSE
uniform sampler2D _t02; // aiTextureType_SPECULAR
uniform sampler2D _t03; // aiTextureType_AMBIENT
uniform sampler2D _t04; // aiTextureType_EMISSIVE
uniform sampler2D _t05; // aiTextureType_HEIGHT
uniform sampler2D _t06; // aiTextureType_NORMALS
uniform sampler2D _t07; // aiTextureType_SHININESS
uniform sampler2D _t08; // aiTextureType_OPACITY
uniform sampler2D _t09; // aiTextureType_DISPLACEMENT
uniform sampler2D _t10; // aiTextureType_LIGHTMAP
uniform sampler2D _t11; // aiTextureType_REFLECTION
uniform sampler2D _t12; // aiTextureType_BASE_COLOR
uniform sampler2D _t13; // aiTextureType_NORMAL_CAMERA
uniform sampler2D _t14; // aiTextureType_EMISSION_COLOR
uniform sampler2D _t15; // aiTextureType_METALNESS
uniform sampler2D _t16; // aiTextureType_DIFFUSE_ROUGHNESS
uniform sampler2D _t17; // aiTextureType_AMBIENT_OCCLUSION
uniform sampler2D _t18; // aiTextureType_SHEEN
uniform sampler2D _t19; // aiTextureType_CLEARCOAT
uniform sampler2D _t20; // aiTextureType_TRANSMISSION

uniform vec3 light_position;
uniform vec3 light_color;
uniform vec3 view_p;
uniform float specular_strength;
uniform float light_intensity;

#define AI_TEXTURE_TYPE_MAX 21
uniform vec4 material_colors[AI_TEXTURE_TYPE_MAX + 1];

void main() {
  float specular_ = texture2D(_t02, tex_coord).r;

  vec4 albedo;
  albedo = texture2D(_t01, tex_coord);
  albedo *= vcolor;
  albedo.rgb = max(albedo.rgb, vec3(0.05));
  
  vec3 adjusted_light_color = light_color * light_intensity;
  float ambient = 0.4;
  vec3 norm = normalize(v_normal);
  vec3 light_dir = normalize(light_position - v_pos);
  float diff = max(dot(norm, light_dir), 0.0);
  float diffuse = diff;
  vec3 view_dir = normalize(view_p - v_pos);
  vec3 reflect_dir = reflect(-light_dir, norm);
  float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
  vec3 specular = specular_ * spec * adjusted_light_color;
  vec3 brightness_magic = vec3(0.2126, 0.7152, 0.0722);
  vec3 albedo_multiply = albedo.rgb * brightness_magic;
  float albedo_brightness = albedo_multiply.x + albedo_multiply.y + albedo_multiply.z;
  //float albedo_brightness = max(max(albedo.r, albedo.g), albedo.b);
  //vec3 final_color = albedo.rgb * max(diffuse, ambient);
  float lowest = min(min(light_color.r, light_color.g), light_color.b);
  vec3 final_color = albedo.rgb * lowest + (albedo_brightness * light_color * (1.0 - lowest)) + specular;
  gl_FragColor = vec4(final_color, albedo.a);
}
