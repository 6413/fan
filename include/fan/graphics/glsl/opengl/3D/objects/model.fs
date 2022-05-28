R"(

#version 130

out vec4 color;

in vec3 fragment_position;
in vec2 texture_coordinate;
in vec3 normal;
in float is_light;
in vec3 view_position;
in vec3 tanget_fragment_position;
in vec3 tanget_view_position;
in vec3 tanget_light_position;

uniform sampler2D texture_diffuse;
uniform sampler2D texture_depth;
uniform samplerCube skybox;

uniform float s;

void main() {

  vec4 object_color = vec4(1, 1, 1, 1);


  vec3 light_color = vec3(0.5, 0, 0);

  vec3 ambient = 0.1 * light_color;

  vec4 diffuse_map = texture(texture_diffuse, texture_coordinate);

  vec3 texture_normal = texture(texture_depth, texture_coordinate).xyz;
  texture_normal = normalize(texture_normal * 2 - 1.0); 

  vec3 lightDir = normalize(tanget_light_position - tanget_fragment_position);
  float diff = max(dot(lightDir, texture_normal), 0.0);
  vec3 diffuse = diff * diffuse_map.xyz;
  // specular
  vec3 viewDir = normalize(tanget_view_position - tanget_fragment_position);
  vec3 halfwayDir = normalize(lightDir + viewDir);  
  float spec = pow(max(dot(texture_normal, halfwayDir), 0.0), 32.0);

  vec3 specular = vec3(0.5) * spec;

  if (s > 0) {
    color = vec4(ambient + diffuse + specular, diffuse_map.w);
  }
  else {
    vec3 i = normalize(fragment_position - view_position);
    vec3 r = reflect(i, normalize(normal));
    color = vec4(texture(skybox, r).rgb * vec3(1, 0.3, 0.3), diffuse_map.w);
  }
}

)"