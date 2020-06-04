#version 330 core

in vec2 texture_coordinates;
in vec3 normal;
in vec3 fragment_position;

uniform vec3 light_position; 
uniform vec3 view_position; 
uniform vec3 light_color;

out vec4 color;
in float visibility;

uniform vec3 sky_color;
uniform sampler2D texture_sampler;

void main() {
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * light_color;

    vec3 norm = normalize(normal);
    vec3 light_direction = normalize(light_position - fragment_position);
    float diff = max(dot(norm, light_direction), 0.0);
    vec3 diffuse = diff * light_color;
    
    float specularStrength = 0.5;
    vec3 view_direction = normalize(view_position - fragment_position);
    vec3 reflect_direction = reflect(-light_direction, norm);  
    float spec = pow(max(dot(view_direction, reflect_direction), 0.0), 32);
    vec3 specular = specularStrength * spec * light_color;  

    vec3 result = (ambient + diffuse + specular) * texture(texture_sampler, texture_coordinates).xyz;

    color = vec4(result, 1);
   // color = mix(vec4(sky_color, 1), color, visibility);
}