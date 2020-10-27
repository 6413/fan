#version 330 core
out vec4 color;

in vec3 normals;  
in vec3 fragment_position;  
  
uniform vec3 light_position; 
uniform vec3 view_position;

in vec2 texture_coordinate;

uniform sampler2D texture1;

void main()
{
    vec3 light_color = vec3(1, 1, 1);
    vec3 objectColor = texture(texture1, texture_coordinate).xyz;
    // ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * light_color;
  	
    // diffuse
    vec3 norm = normalize(normals);
    vec3 lightDir = normalize(light_position - fragment_position);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * light_color;
    
    // specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(view_position - fragment_position);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * light_color;  
        
    vec3 result = (ambient + diffuse + specular) * objectColor;
    color = vec4(result, 1);
    //if (id == someid) {
    //  color = vec4(0, 1, 0, 1);
    //}
} 