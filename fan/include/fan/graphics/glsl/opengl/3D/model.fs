R"(

#version 130

out vec4 color;

in vec3 p;
in vec3 normal;

void main() {

  vec4 object_color = vec4(1, 0, 0, 1);

  vec3 light_position = vec3(0, 10, 10);
  vec3 light_color = vec3(1, 1, 1);

  vec3 norm = normalize(normal);
  vec3 lightDir = normalize(light_position - p);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = diff * light_color;

  color = vec4(object_color.rgb * diffuse, object_color.a);
}

)"