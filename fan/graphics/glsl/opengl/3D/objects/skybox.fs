R"(
#version 130
out vec4 color;

in vec3 texture_coordinate;

uniform samplerCube skybox;

void main()
{    
  color = texture(skybox, texture_coordinate) * vec4(1, 0.3, 0.3, 1);
}

)"