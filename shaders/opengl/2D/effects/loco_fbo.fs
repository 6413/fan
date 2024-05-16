#version 330

in vec2 texture_coordinate;

layout (location = 0) out vec4 o_attachment0;

uniform sampler2D _t00; // HDR color texture
uniform sampler2D _t01; // Bloom texture
uniform sampler2D _t02; // Bloom texture
uniform float bloom_strength = 0;
uniform float gamma = 1.03;
uniform float gamma2 = 1.03;
uniform float bloom_gamma = 1.03;
uniform float exposure = 1.0;

vec3 apply_bloom(vec3 hdrColor, vec3 bloomColor)
{
    // Calculate brightness
    float brightness = dot(bloomColor, vec3(0.2126, 0.7152, 0.0722));

    // Calculate the bloom intensity
    float bloomIntensity = smoothstep(0.6, 1.0, bloom_strength * brightness);

    hdrColor = pow(hdrColor, vec3(1.0 / gamma));
    bloomColor = pow(bloomColor, vec3(1.0 / bloom_gamma));

    // Mix the hdrColor and bloomColor based on the bloom intensity
    //vec3 result = mix(hdrColor, bloomColor, bloom_strength);
    vec3 result = hdrColor + bloomColor * bloomIntensity;

    // Apply exposure and gamma correction
    result = vec3(1.0) - exp(-result * exposure);
    result = pow(result, vec3(1.0 / gamma2));

    return result;
}

void main() {
    vec3 hdrColor = texture(_t00, texture_coordinate).rgb;
    vec3 bloomColor = texture(_t01, texture_coordinate).rgb;

    // Apply bloom effect
    vec3 finalColor = apply_bloom(hdrColor, bloomColor);

    o_attachment0 = vec4(finalColor, 1.0);
}
