#version 430 core
layout (location = 0) in vec3 vertices;
layout (location = 3) in mat4 model;

out vec2 texture_coordinate;

uniform mat4 view;
uniform mat4 projection;

layout(std430, binding = 0) buffer texture_coordinate_layout 
{
    vec2 texture_coordinates[];
};

layout(std430, binding = 1) buffer textures_layout
{
    int textures[];
};


void main() {
    int index = gl_VertexID + textures[gl_InstanceID] * 36;
    texture_coordinate = texture_coordinates[index];
   
    gl_Position = projection * view * model * vec4(vertices, 1.0);
}