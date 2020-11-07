#version 410 core

layout (location = 1) in vec3 vertices;
layout (location = 2) in vec2 texture_coordinates;
layout (location = 3) in vec3 normal;

uniform mat4 projection;
uniform mat4 view;

out vec2 texture_coordinate;  

out vec3 normals;
out vec3 fragment_position;

void main() {
   // mat4 model = mat4(1);
   // model =  mat4(1.0, 0.0, 0.0, vertices[0], 
   //               0.0, 1.0, 0.0, vertices[1], 
   //               0.0, 0.0, 1.0,  vertices[2],  
   //               0.0, 0.0, 0.0,  1.0);
  //  normals = cross(;
    normals = normal;
    texture_coordinate = texture_coordinates;
    gl_Position = projection * view * vec4(vertices, 1.0);
    fragment_position = vertices;
}