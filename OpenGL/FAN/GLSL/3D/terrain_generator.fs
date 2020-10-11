#version 410 core

in vec4 t_color;

in vec2 texture_coordinate;

out vec4 shape_color;

in vec3 fragment_position;
in vec3 normals;

uniform vec3 light_position;
uniform sampler2D texture1;

void main() {
    vec3 view_position = light_position - vec3(0, -2, 0);
    vec3 light_color = vec3(1.f, 1.f, 1.f);
    float ambientStrength = 0.5;
    vec3 ambient = ambientStrength * light_color;

    vec3 norm = normalize(normals);
    vec3 light_direction = normalize(light_position - fragment_position);
    float diff = max(dot(norm, light_direction), 0.0);
    vec3 diffuse = diff * light_color;
    
    float specularStrength = 0.5;
    vec3 view_direction = normalize(view_position - fragment_position);
    vec3 reflect_direction = reflect(-light_direction, norm);  
    float spec = pow(max(dot(view_direction, reflect_direction), 0.0), 32);
    vec3 specular = specularStrength * spec * light_color;  

    vec3 result = (ambient + diffuse + specular) * texture(texture1, texture_coordinate).xyz;

    shape_color = vec4(result, 1);
} 