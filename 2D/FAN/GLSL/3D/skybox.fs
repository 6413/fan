#version 330 core
in vec3 texture_coordinates;
out vec4 color;

uniform samplerCube skybox;
uniform vec3 fog_color;

const float lower_limit = 0;
const float upper_limit = 30;

void main() {
    color = texture(skybox, texture_coordinates);
    
   // float factor = (texture_coordinates.y - lower_limit) / (upper_limit - lower_limit);
   // factor = clamp(factor, 0, 1) * 50;
    //color = mix(vec4(fog_color, 1), final_color, factor);
}