#version 330 core

in vec2 texture_coordinate;
layout (location = 0) out vec4 o_attachment0;

uniform sampler2D _t00;
uniform sampler2D _t01;
uniform float _time;
uniform float bloom_strength = 0.004;
uniform float gamma = 2.2f;
uniform float edge0 = 0.3f;
uniform float edge1 = 1.0f;
uniform float exposure = 1.0f;

float iTime = _time;


vec3 apply_bloom(vec3 hdrColor, vec3 bloomColor)
{
    // Calculate brightness
    float brightness = dot(bloomColor, vec3(0.2126, 0.7152, 0.0722));

    // Calculate the smoothstep value
    float t = smoothstep(edge0, edge1, bloom_strength * brightness);

    // Mix the hdrColor and bloomColor based on the smoothstep value
    vec3 result = mix(hdrColor, bloomColor, t);

    // Apply exposure and gamma correction
    result = vec3(1.0) - exp(-result * exposure);
    result = pow(result, vec3(1.0 / gamma));

    return result;
}

void main() {
  vec3 hdrColor = texture(_t00, texture_coordinate).rgb;
  vec3 bloomColor = texture(_t01, texture_coordinate).rgb;
  //o_attachment0 = vec4(apply_bloom(hdrColor, bloomColor), 1.0);
  o_attachment0 = vec4(hdrColor, 1.0);
}
