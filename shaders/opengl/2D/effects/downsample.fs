#version 330 core

layout (location = 0) out vec3 o_color;

in vec2 texture_coordinate;

uniform sampler2D _t00;
uniform vec2 resolution;
uniform int mipLevel = 1;

vec3 PowVec3(vec3 v, float p) {
  return vec3(pow(v.x, p), pow(v.y, p), pow(v.z, p));
}

const float invGamma = 1.0 / 2.2;
vec3 ToSRGB(vec3 v) { return PowVec3(v, invGamma); }

float sRGBToLuma(vec3 col) {
  return dot(col, vec3(0.2126, 0.7152, 0.0722));
}

float KarisAverage(vec3 col) {
  float luma = sRGBToLuma(ToSRGB(col)) * 0.25;
  return 1.0 / (1.0 + luma);
}

void main() {
  vec2 texelSize = 1.0 / resolution;
    
  if (mipLevel == 0) {
    vec3 s0 = texture(_t00, texture_coordinate + vec2(-1, -1) * texelSize).rgb;
    vec3 s1 = texture(_t00, texture_coordinate + vec2(1, -1) * texelSize).rgb;
    vec3 s2 = texture(_t00, texture_coordinate + vec2(-1, 1) * texelSize).rgb;
    vec3 s3 = texture(_t00, texture_coordinate + vec2(1, 1) * texelSize).rgb;
        
    float w0 = KarisAverage(s0);
    float w1 = KarisAverage(s1);
    float w2 = KarisAverage(s2);
    float w3 = KarisAverage(s3);
        
    float totalWeight = w0 + w1 + w2 + w3;
    vec3 col = (s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3) / totalWeight;

    float brightness = dot(col, vec3(0.2126, 0.7152, 0.0722));
    
    // Non-HDR settings
    float bloom_threshold = 0.5;  // Bloom anything brighter than 50%
    float bloom_softness = 0.2;
    
    float mask = smoothstep(bloom_threshold - bloom_softness, bloom_threshold + bloom_softness, brightness);
    
    o_color = col * mask;
  } 
  else {
    vec3 s0 = texture(_t00, texture_coordinate + vec2(-0.5, -0.5) * texelSize).rgb;
    vec3 s1 = texture(_t00, texture_coordinate + vec2(0.5, -0.5) * texelSize).rgb;
    vec3 s2 = texture(_t00, texture_coordinate + vec2(-0.5, 0.5) * texelSize).rgb;
    vec3 s3 = texture(_t00, texture_coordinate + vec2(0.5, 0.5) * texelSize).rgb;
        
    o_color = (s0 + s1 + s2 + s3) * 0.25;
  }
    
  o_color = max(o_color, 0.0001);
}
