R"(
#version 130

in vec4 color;

out vec4 r_color;

in vec2 f_position;

in vec2 light_size;

uniform vec2 light_position;

uniform float angle;

uniform bool enable_lighting;

uniform float world_light_strength;

void main()
{

//	float ambientStrength = 0.3;
//	vec3 ambient = ambientStrength * color.xyz;

//	vec3 norm = normalize(vec3(1, 0, 0));
//	vec3 lightDir = normalize(vec3(light_position - f_position, 0));
//	float diff = max(dot(norm, lightDir), 0.0);
//	vec3 diffuse = diff * color.xyz;

//	float distance = sqrt(length(light_position - f_position)) * 30;
//	float attenuation = 1.0 / (1 + 0.009 * distance + 0.001 * (distance * distance));    

//	attenuation *= 100;

////diff *= intensity;

//	//diff *= attenuation;
//	ambient *= attenuation;

//	vec3 final_color = (ambient);

//	float theta = dot(lightDir, normalize(-vec3(cos(radians(angle)), -sin(radians(angle)), 0))); 

//	float cutOff = cos(radians(0.1));
//	float outerCutOff = cos(radians(0.0));

//	float epsilon = (cutOff - outerCutOff);

//	float intensity = clamp((theta - outerCutOff) / epsilon, 0.0, 1.0);

    //volatile float x = world_light_strength * 2;

    if (enable_lighting) {
        float distance = length(light_position - f_position);

        float alpha = smoothstep(1.0, 0.1, distance / (light_size.x / 1.8));

        r_color = vec4(color.rgb, alpha / 5);
    }
    else {
        r_color = color;
    }


}
)"