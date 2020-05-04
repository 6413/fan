#version 330 core

in vec2 texture_coordinate;
out vec4 ShapeColor;

uniform sampler2D texture_sampler;

void main() {
  // for (int i = 0; i < 2; i++) {
  //    vec4 first = texture2D(texture_sampler[i], texture_coordinate);
     
 //  }
    ShapeColor = texture2D(texture_sampler, texture_coordinate);
} 