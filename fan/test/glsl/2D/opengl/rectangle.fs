#version 130

in vec4 color;

out vec4 r_color;

in vec2 f_position;

in vec2 light_position;
in vec4 light_color;
in float light_brightness;
in float light_angle;

uniform bool enable_lighting;

uniform float world_light_strength;


void main()
{
    if (enable_lighting) {
        vec3 ambient = world_light_strength * color.xyz;

        float distance = sqrt(length(light_position - f_position)) / light_brightness;
        float attenuation = 1.0 / (1 + 0.009 * distance + 0.001 * (distance * distance));    

        attenuation *= 100;

        ambient *= attenuation;

        vec4 final_color = vec4(ambient, 1);

        r_color = mix(final_color, light_color * world_light_strength * attenuation, 0.5);
    }
    else {
        r_color = color;
    }


} 